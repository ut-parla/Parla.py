import csv
import time
import argparse

import threading

from sleep.core import bsleep, sleep_with_gil

free_sleep = bsleep
lock_sleep = sleep_with_gil

parser = argparse.ArgumentParser()
parser.add_argument("-workers", type=int, default=1)
parser.add_argument("-n", type=int, default=3)
parser.add_argument("-t", type=int, default=10)
parser.add_argument("-accesses", type=int, default=10)
parser.add_argument("-frac", type=float, default=0)
parser.add_argument('-sweep', type=int, default=0)
parser.add_argument('-barrier', type=int, default=0)
args = parser.parse_args()


def task(barrier, n, t, accesses, frac, free_time, lock_time):
    for k in range(accesses):
        free_sleep(free_time)
        lock_sleep(lock_time)

def thread_task(barrier, n, t, accesses, frac):

    kernel_time = t / accesses
    free_time = kernel_time * (1 - frac)
    lock_time = kernel_time * frac
    for i in range(n):
        task(barrier, n, t, accesses, frac, free_time, lock_time)
        if args.barrier:
            barrier.wait()

def drange(start, stop):
    while start < stop:
        yield start
        start <<= 1

def main(workers, n, task_time, accesses, frac):
    barrier = threading.Barrier(workers)
    threads = []

    for i in range(workers):
        per = (n//workers + (i < n % workers))
        t = threading.Thread(target=thread_task, args=(barrier, per,
                                                       task_time, accesses, frac))
        threads.append(t)

    start_t = time.perf_counter()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    end_t = time.perf_counter()

    elapsed_t = end_t - start_t
    print(', '.join([str(workers), str(n), str(task_time), str(
        accesses), str(frac), str(elapsed_t)]), flush=True)

if __name__ == "__main__":
    print(', '.join([str('workers'), str('n'), str('task_time'), str(
        'accesses'), str('frac'), str('total_time')]), flush=True)
    for task_time in [1000, 5000, 10000, 50000, 100000]:
        for accesses in [1, 10]:
            for nworkers in drange(1, args.workers):
                for frac in [0]:
                    main(nworkers, args.n, task_time, accesses, frac)




