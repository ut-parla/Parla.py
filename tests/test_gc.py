# Import Parla
from parla import Parla, spawn, parray, TaskSpace
# Import the 'cpu' device type
from parla.cpu import cpu
from parla.cuda import gpu

import parla.tracking 

import numpy as np

import time

def main():
    A = parray.asarray(np.random.rand(10000, 10000))
    B = parray.asarray(np.random.rand(10000, 10000))
    t = TaskSpace("AxB")

    for i in range(0, 4):
        # Spawn a task to be scheduled by the Parla runtime
        """
        @spawn(t[i], input=[A, B], placement=gpu(i))
        def axb():
            C = A @ B
            print("axb is called", flush=True)
            time.sleep(10)
        """
        @spawn(t[i], input=[A], placement=gpu(i))
        def axb():
            print("axb is called", flush=True)
            time.sleep(3)

    @spawn(t[4], dependencies=[t[:4]])
    def apb():
        print("apb", flush=True)
        time.sleep(2)

    @spawn(dependencies=[t[4]], input=[A], placement=gpu(1))
    def apb():
        print("apb2", flush=True)
        print(A.array)
        time.sleep(2)



# Execute the Parla program within the Parla context
if __name__ == "__main__":
    with Parla():
        main()
