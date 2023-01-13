import argparse
import numpy as np
import numba
from numba import njit, jit

import math
import time

from parla import Parla, get_all_devices
from parla.cpu import cpu

from parla.tasks import spawn, TaskSpace, tasks

from sleep.core import bsleep, sleep_with_gil

free_sleep = bsleep
lock_sleep = sleep_with_gil


def waste_time(free_time, gil_time, accesses):
    for k in range(accesses):
        free_sleep(free_time)
        lock_sleep(gil_time)


@jit(nogil=True, parallel=False)
def increment(array, counter):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] += np.random.rand()


def increment_wrapper(array, counter):
    increment(array, counter)


def main(N, d, steps, NUM_WORKERS, cpu_array, sync_flag, vcu_flag,
         dep_flag, verbose, sleep_time, accesses, gil_fraction,
         sleep_flag, strong_flag, restrict_flag):

    @spawn(vcus=0)
    async def main_task():

        T = TaskSpace("Outer")

        start_t = time.perf_counter()
        for t in range(steps):

            for ng in range(NUM_WORKERS):

                if not dep_flag or (t == 0):
                    deps = []
                else:
                    deps = [T[1, t-1, ng]]

                vcus = 1.0/NUM_WORKERS

                kernel_time = sleep_time / accesses

                if strong_flag:
                    kernel_time = kernel_time / NUM_WORKERS

                free_time = kernel_time * (1.0 - gil_fraction)
                lock_time = kernel_time * gil_fraction

                @spawn(T[1, t, ng], dependencies=deps, vcus=vcus)
                def task():
                    if verbose:
                        print("Task", [1, t, ng], "Started.", flush=True)
                        inner_start_t = time.perf_counter()

                    if sleep_flag:
                        waste_time(free_time, lock_time, accesses)
                    else:
                        array = cpu_array[ng]
                        increment_wrapper(array, 100000)

                    if verbose:
                        inner_end_t = time.perf_counter()
                        inner_elapsed = inner_end_t - inner_start_t

                        print("Task", [1, t, ng], "Finished. I took ",
                              inner_elapsed, flush=True)
                        #print("I am task", [1, t, ng], ". I took ", inner_elapsed, ". on device", A.device, flush=True)

            if sync_flag:
                if restrict_flag:
                    await T[1, t, :]
                else:
                    await T

        if not sync_flag:
            if restrict_flag:
                await T[1, steps-1, :]
            else:
                await T
        end_t = time.perf_counter()

        elapsed = end_t - start_t
        print(', '.join([str(NUM_WORKERS), str(steps), str(sleep_time),
              str(accesses), str(gil_fraction), str(elapsed)]), flush=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('-d', type=int, default=7)
    parser.add_argument('-n', type=int, default=2**23)
    parser.add_argument('--isync', type=int, default=0)
    parser.add_argument('--vcus', type=int, default=0)
    parser.add_argument('--deps', type=int, default=1)
    parser.add_argument('--verbose', type=int, default=0)

    parser.add_argument("-t", type=int, default=10)
    parser.add_argument("--accesses", type=int, default=10)
    parser.add_argument("--frac", type=float, default=0)

    parser.add_argument('--strong', type=int, default=0)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--restrict', type=int, default=0)

    args = parser.parse_args()

    NUM_WORKERS = args.workers
    STEPS = args.steps
    N = args.n
    d = args.d
    isync = args.isync

    if args.strong:
        N = N//NUM_WORKERS

    cpu_array = []
    for ng in range(NUM_WORKERS):
        # cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
        cpu_array.append(np.zeros([N, d]))
        increment_wrapper(cpu_array[ng], 1)

    with Parla():
        main(N, d, STEPS, NUM_WORKERS, cpu_array, isync, args.vcus,
             args.deps, args.verbose, args.t, args.accesses, args.frac,
             args.sleep, args.strong, args.restrict)
