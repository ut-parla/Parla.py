import numpy as np

from parla import Parla
from parla.cpu import cpu
from parla.tasks import spawn, Req
from numba import jit

@jit
def subdivide(array, split):
    low = 0
    high = array.shape[0] - 1
    while True:
        while array[low] <= split:
            low += 1
        if low > high:
            return low
        while array[high] > split:
            if low >= high:
                return low
            high -= 1
        array[low], array[high] = array[high], array[low]
        low += 1
        if low >= high:
            return low
        high -= 1

# Insertion sort for non-recursive base case
@jit
def insertion_sort(array):
    for i in range(array.shape[0]):
        j = i
        while j > 0 and array[j-1] > array[j]:
            array[j-1], array[j] = array[j], array[j-1]
            j -= 1

class qsort_bin_step:
    def __init__(self, array, lower, upper, threshold):
        self.array = array
        self.lower = lower
        self.upper = upper
        self.threshold = threshold
    def __call__(self):
        if self.array.shape[0] < self.threshold:
            insertion_sort(self.array)
        else:
            split = .5 * (self.lower + self.upper)
            split_idx = subdivide(self.array, split)
            lower_array = self.array[:split_idx]
            upper_array = self.array[split_idx:]
            lower_step = qsort_bin_step(lower_array, self.lower, split, self.threshold)
            @spawn(placement = cpu)
            def lower_step_task():
                lower_step()
            upper_step = qsort_bin_step(upper_array, split, self.upper, self.threshold)
            @spawn(placement = cpu)
            def upper_step_task():
                upper_step()

def quicksort(array, lower = 0., upper = 1., threshold = 100):
    with Parla():
        step = qsort_bin_step(array, lower, upper, threshold)
        @spawn(placement = cpu)
        def step_task():
            step()

a = np.random.rand(10000)
a_sorted = np.sort(a)
quicksort(a)
assert (a == a_sorted).all()
