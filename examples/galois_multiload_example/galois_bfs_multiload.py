from threading import Thread
import parla.vec_backtrace
from parla.multiload import multiload, multiload_contexts, mark_module_as_global

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input graph")
parser.add_argument("vecs", help="# of threads/vecs to run", type=int)
args = parser.parse_args()

#set cpu must be before loading galois
multiload_contexts[0].set_allowed_cpus([0,1])

with multiload():
    import parla_galois.core

def bfs_sssp(i):
    with multiload_contexts[i]:
        from time import sleep
        #import parla_galois.core
        parla_galois.core.py_init_galois()
        g = parla_galois.core.py_load_file(args.input)
        #source = i
        #report = (i+1)*5
        source = 0
        report = 5
        
        slot = i
        parla_galois.core.py_bfs(g, source, slot)
        print(f"distance from {source} to {report} at slot {slot} is {parla_galois.core.py_distance(g, report, slot)}")

        parla_galois.core.py_delete_galois()

bfs_sssp(0)

# threads = []
# for i in range(args.vecs):
#     threads.append(Thread(target=bfs_sssp, args=(i,)))

# #threads.append(Thread(target=bfs_sssp, args=(0,)))

# for t in threads: t.start()
# for t in threads: t.join()
# print("ye")
