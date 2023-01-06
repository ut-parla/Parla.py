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
parser.add_argument('-barrier', type=int, default=1)
args = parser.parse_args()


def thread_task(barrier, n, t, accesses, frac):

    kernel_time = t / accesses
    free_time = kernel_time * (1 - frac)
    lock_time = kernel_time * frac

    for i in range(n):
        for k in range(accesses):
            free_sleep(free_time)
            lock_sleep(lock_time)
        if args.barrier:
            barrier.wait()


if __name__ == "__main__":

    barrier = threading.Barrier(args.workers)
    threads = []

    for i in range(args.workers):
        t = threading.Thread(target=thread_task, args=(barrier, args.n,
                                                       args.t/args.workers, args.accesses, args.frac))
        threads.append(t)

    start_t = time.perf_counter()
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    end_t = time.perf_counter()

    print(end_t - start_t)



