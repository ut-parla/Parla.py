"""
A implementation of parallel k-means (init by k-means++) on CPU/GPU.
"""

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu, summarize_memory
from parla.tasks import spawn, TaskSpace
import numpy as np
import cupy as cp
from time import perf_counter
import sys
from parla.parray import asarray_batch


DEBUG = False

def vector_str(vector: list):
    return "[" + ", ".join(map(str, vector)) + "]"


def check_stop_criteria(new_centroids, centroids, K):
    """
    Return true if stop criteria is met
    """
    threshold = 0.000000000001

    if True in cp.linalg.norm(new_centroids - centroids, axis=1) > threshold:
        return False

    return True


async def kmeanspp(data: np.ndarray, K: int, centroids: np.ndarray):
    """
    Parallel k-means++ on CPU.
    This algorithm will choosing the initial values for the k-means clustering.

    centroids: the return value, will be modified
    """
    N, D = data.shape

    task = TaskSpace("KmeansppTask")
    task_counter = 0

    distance = np.full((K, N), np.Inf)

    print(f"Begins with dataset with {N} points, need to find {K} centroids")

    # the first centroid is chose randomly
    centroids[0] = data[np.random.randint(0, N - 1)]
    if DEBUG:
        print(f"Select centroid 0 randomly: {vector_str(centroids[0])}")

    for i in range(K - 1):
        # compute distance to each centroid separately
        for j in range(i + 1):
            # Fact: it is faster if iteration is used rather than Parla tasks
            @spawn(task[task_counter], placement=cpu)
            async def update_distance():
                distance[j] = np.linalg.norm(data - centroids[j], axis=1)

            task_counter += 1
        await task

        # for each point find its distance to nearest centroid
        minimum_distance = np.amin(distance, axis=0)

        # normalized the distance to generate a distribution
        normalized_distribution = minimum_distance / np.sum(minimum_distance)

        # select next centroid follow the distribution
        next_centroid_idx = np.random.choice(range(N), p=normalized_distribution)
        centroids[i + 1] = data[next_centroid_idx]

        if DEBUG:
            print(f"Select centroid {i + 1}: {vector_str(centroids[i + 1])}")

    print("K-means++ done.")


def kmeans(data: cp.ndarray, K: int, centroids: cp.ndarray):
    """
    Clusters points into k clusters using k_means clustering.
    """
    print("Start K-means clustering.")
    N, D = data.shape
    new_centroids = cp.full((K, D), 0.0)
    loop = 1
    while loop < 200:
        # assign each point to nearest cluster
        distance = cp.full((K, N), 0.0)
        for centroid_idx in range(K):
            # NEW: extract the underlying array when use static method from cupy
            distance[centroid_idx] = cp.linalg.norm(data - centroids[centroid_idx], axis=1)
        assignment = cp.argmin(distance, axis=0)


        for i in range(K):
            condition = assignment == i
            # build new clusters
            cluster = data[condition]
            # compute new centroids
            if cluster.size != 0:
                new_centroids[i] = cp.mean(cluster, axis=0)

        # stop when the distance of current centroids to last centroids are lower than threshold
        if check_stop_criteria(new_centroids, centroids, K):
            pass  # change this to 'pass' will let the loops has a fixed number of iteration

        loop += 1

        # NEW: keep the reference unchanged
        # LHS: PArray / RHS: return a numpy.ndarray
        # centroids = np.copy(new_centroids)
        centroids[:] = np.copy(new_centroids)

    print(f"K-means done with {loop} loops.")

    if DEBUG:
        for k in range(K):
            print(f"Predicted Centroid {k}: {vector_str(centroids[k])}")


def generate_random_dataset(K, D, N):
    """
    Generate a random dataset with N points,
    which distributed uniformly to K clusters at D dimensional (make it easier to validate centroids).
    return a N X D matrix
    """
    range_begin = 0.0
    range_end = 1000.0
    distance_to_center = 100.0

    num_points_of_each_cluster = N // K
    points_left = N % K

    if DEBUG:
        data = np.full((N, D), 0.0)
    else:
        data = np.random.rand(N, D)

    if DEBUG:
        for k in range(K):
            begin = num_points_of_each_cluster * k
            end = num_points_of_each_cluster * (k + 1)
            if k == K - 1:
                end += points_left


            center_list = []

            for d in range(D):
                center = np.random.uniform(range_begin, range_end)
                center_list.append(str(center))
                for i in range(begin, end):
                    data[i][d] = center + np.random.uniform(-distance_to_center, distance_to_center)

            center_str = ", ".join(center_list)
            print(f"Cluster {k} center at [{center_str}]")

    print(f"Generated a random dataset with {N} points to {K} clusters at {D} dimensional")
    return data


def parse_input_file(file_path):
    """
    Get dataset from a file
    format:
    N
    1 <value1> <value2> ... \n
    2 <value1> <value2> ... \n
    ...
    """
    with open(file_path, "r") as in_file:
        lines = in_file.readlines()

        N = int(lines[0])
        D = len(lines[1].split()) - 1

        data = np.full((N, D), 0.0)
        for i in range(N):
            values = lines[i+1].split()
            for j in range(D):
                data[i][j] = float(values[j+1])

    return data


async def start(data, K, task):

    D = data.shape[1]
    centroids = np.full((K, D), 0.0)

    # # 1. NEW: convert to PArray
    # data, centroids = asarray_batch(data, centroids)

    # 2. NEW: input/output
    @spawn(task[0], placement=cpu)
    async def start_kmeanspp():
        kmeanspp_timer_begin = perf_counter()

        await kmeanspp(data, K, centroids)

        kmeanspp_timer_end = perf_counter()
        print(f"K-Means plus plus takes {kmeanspp_timer_end - kmeanspp_timer_begin} seconds")

    # 3. NEW: input/inout
    @spawn(task[1], [task[0]], placement=gpu)
    async def start_kmeans():
        nonlocal data, centroids
        copy_timer_begin = perf_counter()

        # 4. NEW: no need to copy data to device manually
        data = cp.asarray(data)
        centroids = cp.asarray(centroids)

        copy_timer_end = perf_counter()
        print(f"Copy data from host to device takes {copy_timer_end - copy_timer_begin} seconds")

        # run kmeans
        kmeanspp_timer_begin = perf_counter()

        kmeans(data, K, centroids)

        kmeanspp_timer_end = perf_counter()
        print(f"K-Means Clustering takes {kmeanspp_timer_end - kmeanspp_timer_begin} seconds")


def main():
    """
    Launch tasks.
    """
    task = TaskSpace("Dummy")
    @spawn(task[0])
    async def main_task():
        np.random.seed(30)
        generate_timer_begin = perf_counter()

        if len(sys.argv) == 1:  # no commend line argument
            K = 10  # number of clusters
            D = 10  # point dimension
            N = 10000000 # number of points
            data = generate_random_dataset(K, D, N)
        elif len(sys.argv) == 3:  # the first argument is the input file
            file_path = sys.argv[1]
            K = int(sys.argv[2])
            data = parse_input_file(file_path)
        else:
            raise Exception("Invalid arguments")

        generate_timer_end = perf_counter()
        print(f"Generate dataset takes {generate_timer_end - generate_timer_begin} seconds")

        main_timer_begin = perf_counter()

        ts = TaskSpace("Task")
        await start(data, K, ts)

        await ts
        main_timer_end = perf_counter()
        print(f"The whole program takes {main_timer_end - main_timer_begin} seconds")


if __name__ == "__main__":
    with Parla():
        main()
    summarize_memory()
