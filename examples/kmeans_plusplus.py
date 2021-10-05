"""
A implementation of parallel k-means (init by k-means++) on CPU/GPU.
"""

from parla import Parla
from parla.cpu import cpu
from parla.cuda import gpu
from parla.tasks import spawn, TaskSpace
import numpy as np
import cupy as cp
import random


async def kmeanspp(data: np.ndarray, k: int, centroids: list):
    """
    Parallel k-means++ on CPU.
    This algorithm will choosing the initial values for the k-means clustering.

    centroids: the return value, will be modified
    """
    task = TaskSpace("KmeansppTask")
    task_counter = 0

    num_points = len(data)
    distance = np.full((k, num_points), np.Inf)

    print(f"Begins with dataset with {num_points} points, need to find {k} centroids")

    # the first centroid is chose randomly
    centroids.append(data[random.randint(0, num_points - 1)])
    print(f"Select centroid 0 randomly: [{centroids[0][0]}, {centroids[0][1]}]")

    for i in range(k - 1):
        # compute distance to each centroid separately
        for j in range(i + 1):
            @spawn(task[task_counter])
            async def update_distance():
                for point_idx in range(num_points):
                    distance[j][point_idx] = np.linalg.norm(centroids[j] - data[point_idx])

            task_counter += 1
        await task

        # for each point find its distance to nearest centroid
        minimum_distance = np.amin(distance, axis=0)

        # normalized the distance to generate a distribution
        normalized_distribution = [d / np.sum(minimum_distance) for d in minimum_distance]

        # select next centroid follow the distribution
        next_centroid_idx = np.random.choice(range(num_points), p=normalized_distribution)
        centroids.append(data[next_centroid_idx])

        print(f"Select centroid {i + 1}: [{centroids[i + 1][0]}, {centroids[i + 1][1]}]")

    print("K-means++ done.")


def kmeans(data: cp.ndarray, k: int, centroids: cp.ndarray):
    """
    Clusters points into k clusters using k_means clustering.
    """
    print("Start K-means clustering.")
    num_points = len(data)
    last_assignment = cp.zeros(num_points)

    loop = 1
    while True:
        print(f"Loop {loop}")
        # assign each point to nearest cluster
        distance = cp.zeros((k, num_points))
        for centroid_idx in range(k):
            distance[centroid_idx] = cp.linalg.norm(data - centroids[centroid_idx], axis=1)
        assignment = cp.argmin(distance, axis=0)

        for i in range(k):
            condition = assignment == i
            # build new clusters
            cluster = data[condition]
            # compute new centroids
            if cluster.size != 0:
                centroids[i] = cp.mean(cluster, axis=0)

        # stop when no point change its cluster
        if cp.array_equal(last_assignment, assignment):
            print("K-means Done.")
            return assignment

        last_assignment = cp.copy(assignment)
        loop += 1


def main(data: np.ndarray, k: int):
    """
    Launch tasks.
    """

    centroids = []

    task = TaskSpace("Task")

    @spawn(task[0], placement=cpu)
    async def start_kmeanspp():
        await kmeanspp(data, k, centroids)

    @spawn(task[1], [task[0]], placement=gpu)
    async def start_kmeans():
        assignment = kmeans(cp.asarray(data), k, cp.asarray(centroids))
        print(assignment)


if __name__ == "__main__":
    k = 5  # number of clusters
    num_data = 1000

    # generate a random dataset which distributed uniformly to 5 clusters (make it easier to validate centroids)
    # which is in the space: ([0, 100] & [200, 300]) X ([0, 100] & [200, 300]) plus [100, 200] X [100, 200]
    data = np.array(
        [[random.randint(0, 100) + (random.randint(0, 1) * 200), random.randint(0, 100) + (random.randint(0, 1) * 200)]
         for i in range(num_data // 5 * 4)])
    data = np.append(data, [[random.randint(100, 200), random.randint(100, 200)] for i in range(num_data // 5)], axis=0)

    with Parla():
        main(data, k)
