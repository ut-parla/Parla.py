#!/usr/bin/env python3
from threading import Thread
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input graph")
parser.add_argument("vecs", help="# of threads/vecs to run", type=int)
args = parser.parse_args()

def bfs_sssp(i):
    from parla_galois.core import py_load_file, py_init_galois
    import parla_galois.core
    py_init_galois()
    g = py_load_file(args.input)
    source = i
    report = (i+1)*5
    slot = i
    g = py_load_file(args.input)
    parla_galois.core.py_bfs(g, source, slot)
    print(f"distance from {source} to {report} at slot {slot} is {parla_galois.core.py_distance(g, report, slot)}")

threads = []
for i in range(args.vecs):
    threads.append(Thread(target=bfs_sssp, args=(i,)))

for t in threads: t.start()
for t in threads: t.join()