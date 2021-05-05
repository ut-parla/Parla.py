from threading import Thread, Barrier
import parla.vec_backtrace
from parla.multiload import multiload, multiload_contexts, mark_module_as_global
from timer import Timer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input graph")
parser.add_argument("vecs", help="# of threads/vecs to run", type=int)
parser.add_argument('cores', type=int, nargs="*")
args = parser.parse_args()

# Set cores for each VEC
vecn = 0
for i in range(0, len(args.cores), 2):
    start = args.cores[i]
    end   = args.cores[i+1] 
    if start == end:
        print(f"Skipping set_allowed_cpus for VEC #{vecn}")
    else:
        print(f"Setting set_allowed_cpus for VEC #{vecn} to {start}-{end}")
        multiload_contexts[vecn].set_allowed_cpus([i for i in range(start, end)])
    vecn += 1

# Init Galois on each VEC
with multiload():
    import parla_galois.core

# Function each thread will run
def bfs_sssp(i, barrier):
    #with multiload_contexts[i]:
    from time import sleep
    import parla_galois.core
    with Timer.get_handle("init-galois"):
        parla_galois.core.py_init_galois(len(multiload_contexts[i].allowed_cpus))
    with Timer.get_handle("load-graph"):
        g = parla_galois.core.py_load_file(args.input)
    # sync so everyone runs concurrently
    idx = barrier.wait()
    if idx == 0: barrier.reset()
    
    #source = i
    #report = (i+1)*5
    source = 0
    report = 5
    slot = i
    
    with Timer.get_handle("bfs"):
        parla_galois.core.py_bfs(g, source, slot)
        print(f"distance from {source} to {report} at slot {slot} is {parla_galois.core.py_distance(g, report, slot)}")

    parla_galois.core.py_delete_galois()

#bfs_sssp(0)

barrier = Barrier(args.vecs)
threads = []
for i in range(args.vecs):
    threads.append(Thread(target=bfs_sssp, args=(i,barrier)))

#threads.append(Thread(target=bfs_sssp, args=(0,)))

for t in threads: t.start()
for t in threads: t.join()

Timer.print()
