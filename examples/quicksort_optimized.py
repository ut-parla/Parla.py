import time

import numpy as np

from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn
from numba import jit, njit

@jit("i4(int32[:], i4, i4)", fastmath=True, nogil=True, nopython=True)
def partition(a, lo, hi):
    aa = 0
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] < pivot:
            aa = a[j]
            a[j] = a[i]
            a[i] = aa
            i = i + 1
    aa = a[i]
    a[i] = a[hi]
    a[hi] = aa
    return i

@jit("int32[:](int32[:], i4, i4)", fastmath=True, nogil=True, nopython=True)
def numba_sort(a, lo, hi):
    b = np.sort(a[lo:hi+1])
    return b

@jit("int32[:](int32[:], i4, i4)", fastmath=True, nogil=True, nopython=True)
def insertion_sort(a, lo, hi):
    for i in range(lo, hi+1):
        j = i
        while j > lo and a[j-1] > a[j]:
            aa = a[j-1]
            a[j-1] = a[j]
            a[j] = aa
            j -= 1
    return a

def quicksort(a, lo, hi):
    #print("lo, hi: ", lo, hi, flush=True)

    if lo >= hi - 1000:
        #a[lo:hi+1] = numba_sort(a, lo, hi)
        insertion_sort(a, lo, hi)
        return None

    p = partition(a, lo, hi)

    @spawn(placement=cpu)
    def lower_block_task():
        quicksort(a, lo, p-1)
    @spawn(placement=cpu)
    def upper_block_task():
        quicksort(a, p+1, hi)

if __name__ == '__main__':
    n = 10000000
    a = np.arange(n, dtype=np.int32)
    np.random.shuffle(a)
    a_sorted = np.sort(a)
    start = time.perf_counter()
    with Parla():
        quicksort(a, 0, n-1)
    end = time.perf_counter()
    print(end - start, "seconds")
    assert (a == a_sorted).all()
