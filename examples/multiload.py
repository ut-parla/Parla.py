# TODO: This is more of a test than an example. But it could be made into a useful example after we fix things.

import parla.cpu

import logging

# logging.basicConfig(level=logging.INFO)

from parla.multiload import multiload, multiload_context

with multiload():
    import numpy as np
import timeit

if __name__ == '__main__':
    multiload_context(1).set_allowed_cpus([0])
    multiload_context(2).set_allowed_cpus([1,2,3,4,5,6,7])
    def timed_thing():
        print("np.random.rand =", np.random.rand)
        return timeit.timeit(lambda: np.dot(np.random.rand(2000, 2000), np.random.rand(2000, 2000)), number=1)
    with multiload_context(1):
        ctx_1 = timed_thing()
    with multiload_context(2):
        ctx_2 = timed_thing()
    print(ctx_1, ctx_2)
    # assert ctx_2 > ctx_1*1.5
