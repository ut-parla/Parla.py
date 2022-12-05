import numpy as np
import numba
from numba import njit, jit

import math
import time

from parla import Parla, get_all_devices
from parla.cpu import cpu

from parla.tasks import spawn, TaskSpace

import argparse

@jit(nogil=True)
def increment(array, counter):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] += 1.0


def increment_wrapper(array, counter):
    increment(array, counter)

def main(N, d, steps, NUM_WORKERS, cpu_array, sync_flag, vcu_flag, dep_flag, verbose):

    @spawn(placement=cpu, vcus=0)
    async def main_task():

        T = TaskSpace("Outer")

        start_t = time.perf_counter()
        for t in range(steps):

            for ng in range(NUM_WORKERS):
                loc = cpu

                if dep_flag or (t == 0):
                    deps = []
                else:
                    deps = [T[1, t-1, ng]]

                vcus = 1.0/NUM_WORKERS

                @spawn(T[1, t, ng], dependencies=deps, placement=loc, vcus=vcus)
                def task():
                    if verbose:
                        print("Task", [1, t, ng], "Started.", flush=True)
                        inner_start_t = time.perf_counter()

                    increment_wrapper(cpu_array[ng], 1000)

                    if verbose:
                        inner_end_t = time.perf_counter()
                        inner_elapsed = inner_end_t - inner_start_t

                        print("Task", [1, t, ng], "Finished. I took ", inner_elapsed, flush=True)
                        #print("I am task", [1, t, ng], ". I took ", inner_elapsed, ". on device", A.device, flush=True)

            if sync_flag:
                await T

        if not sync_flag:
            await T
        end_t = time.perf_counter()

        elapsed = end_t - start_t
        print("Elapsed: ", elapsed, flush=True)


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
    args = parser.parse_args()



    NUM_WORKERS = args.workers
    STEPS = args.steps
    N = args.n
    d = args.d
    isync = args.isync

    cpu_array = []
    for ng in range(NUM_WORKERS):
        #cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
        cpu_array.append(np.zeros([N, d]))
        increment_wrapper(cpu_array[ng], 1)

    with Parla():
        main(N, d, STEPS, NUM_WORKERS, cpu_array, isync, args.vcus, args.deps, args.verbose)
