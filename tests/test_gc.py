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
        @spawn(t[i], input=[A, B], placement=gpu(i))
        def axb():
            print(f"{A} >>>> ", flush=True)
            C = A @ B
            print("C is ", type(C))
            time.sleep(10)

    @spawn(dependencies=[t])
    def apb():
        print("apb", flush=True)
        time.sleep(20)



# Execute the Parla program within the Parla context
if __name__ == "__main__":
    with Parla():
        main()
