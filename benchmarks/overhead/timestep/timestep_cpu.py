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


def main(N, d, steps, NUM_WORKERS, WIDTH, cpu_array, sync_flag, vcu_flag,
         dep_flag, verbose, sleep_time, accesses, gil_fraction,
         sleep_flag, strong_flag, restrict_flag):

    @spawn(vcus=0)
    async def main_task():

        T = TaskSpace("Outer")

        start_t = time.perf_counter()
        for t in range(steps):

            if restrict_flag:
                odeps = [T[1, t-1, l] for l in range(WIDTH)]

            for ng in range(WIDTH):

                if not dep_flag or (t == 0):
                    deps = []
                else:
                    if restrict_flag:
                        deps = odeps
                    else:
                        deps = [T[1, t-1, ng]]

                vcus = 1.0/NUM_WORKERS

                kernel_time = sleep_time / accesses

                if strong_flag:
                    kernel_time = kernel_time / NUM_WORKERS
                else:
                    kernel_time = kernel_time

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
                    await tasks([T[1, t, l] for l in range(WIDTH)])
                else:
                    await T

        if not sync_flag:
            if restrict_flag:
                await tasks([T[1, steps-1, l] for l in range(WIDTH)])
            else:
                await T
        end_t = time.perf_counter()

        elapsed = end_t - start_t
        print(', '.join([str(NUM_WORKERS), str(steps), str(sleep_time),
              str(accesses), str(gil_fraction), str(elapsed)]), flush=True)

def drange(start, stop):
    while start < stop:
        yield start
        start <<= 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1, help='How many workers to use. This will perform a sample of 1 to workers by powers of 2')
    parser.add_argument('--width', type=int, default=0, help='The width of the task graph. If not set this is equal to nworkers.')
    parser.add_argument('--steps', type=int, default=1, help='The depth of the task graph.')
    parser.add_argument('-d', type=int, default=7, help='The size of the data if using numba busy kernel')
    parser.add_argument('-n', type=int, default=2**23, help='The size of the data if using numba busy kernel')
    parser.add_argument('--isync', type=int, default=0, help='Whether to synchronize (internally) using await at every timestep.')
    parser.add_argument('--vcus', type=int, default=1, help='Whether tasks use vcus to restrict how many can run on a single device')
    parser.add_argument('--deps', type=int, default=1, help='Whether tasks have dependencies on the prior iteration')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose!')

    parser.add_argument("-t", type=int, default=10, help='The task time in microseconds. These are hardcoded in this main.')
    parser.add_argument("--accesses", type=int, default=10, help='How many times the task stops busy waiting and accesses the GIL')
    parser.add_argument("--frac", type=float, default=0, help='The fraction of the total task time that the GIL is held')

    parser.add_argument('--strong', type=int, default=0, help='Whether to use strong (1) or weak (0) scaling of the task time')
    parser.add_argument('--sleep', type=int, default=1, help='Whether to use the synthetic sleep (1) or the numba busy kernel (0)')
    parser.add_argument('--restrict', type=int, default=0, help='This does two separate things. If using isync it restricts to only waiting on the prior timestep. If using deps, it changes the dependencies from being a separate chain to depending on all tasks in the prior timestep')

    args = parser.parse_args()

    if args.width == 0:
        args.width = args.workers

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

    print(', '.join([str('workers'), str('n'), str('task_time'), str(
        'accesses'), str('frac'), str('total_time')]), flush=True)
    for task_time in [10000, 50000]:
        for accesses in [args.accesses]:
            for nworkers in drange(1, args.workers):
                for frac in [args.frac]:
                    with Parla():
                        main(N, d, STEPS, nworkers, nworkers, cpu_array, isync, args.vcus,
                             args.deps, args.verbose, task_time, accesses, frac,
                             args.sleep, args.strong, args.restrict)
