import csv
import time
import argparse

from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.cpu import cpu

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
parser.add_argument('-verbose', type=int, default=0)
parser.add_argument('-empty', type=int, default=0)
args = parser.parse_args()


def main(workers, n, t, accesses, frac):

    @spawn(vcus=0)
    async def task1():
        cost = 1.0/workers

        kernel_time = t / accesses
        free_time = kernel_time * (1 - frac)
        lock_time = kernel_time * frac

        start_t = time.perf_counter()
        T = TaskSpace("T")

        for i in range(n):
            @spawn(T[i], vcus=cost)
            def task1():
                if args.empty:
                    return None

                if args.verbose:
                    inner_start_t = time.perf_counter()

                for k in range(accesses):
                    free_sleep(free_time)
                    lock_sleep(lock_time)

                if args.verbose:
                    inner_end_t = time.perf_counter()
                    print("Task", i, " | Inner Time: ",
                          inner_end_t - inner_start_t, flush=True)

        await T

        end_t = time.perf_counter()
        elapsed_t = end_t - start_t
        print(', '.join([str(workers), str(n), str(t), str(
            accesses), str(frac), str(elapsed_t)]), flush=True)
        #print(n/elapsed_t, flush=True)

    # @spawn()
    # def test():
    #    print("HELLO", flush=True)

def drange(start, stop):
    while start < stop:
        yield start
        start <<= 1

if __name__ == "__main__":

    print(', '.join([str('workers'), str('n'), str('task_time'), str(
        'accesses'), str('frac'), str('total_time')]), flush=True)
    if not args.sweep:
        with Parla():
            main(args.workers, args.n, args.t, args.accesses, args.frac)
    else:
        for task_time in [1000, 5000, 10000, 50000, 100000]:
            for accesses in [1, 10]:
                for nworkers in drange(1, args.workers):
                    for frac in [0]:
                        with Parla():
                            main(nworkers, args.n, task_time, accesses, frac)
