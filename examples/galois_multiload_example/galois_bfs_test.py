#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input graph")
parser.add_argument("startNode", help="Source node to run bfs", type=int)
parser.add_argument("reportNode", help="Node to report distance from source", type=int)
parser.add_argument("slot", help="Slot that bfs should use, 0 should be fine", type=int)
args = parser.parse_args()

from parla_galois.core import py_load_file, py_init_galois
py_init_galois()
g = py_load_file(args.input)

source = args.startNode
dest = args.reportNode
slot = args.slot

from parla_galois.core import py_bfs, py_distance
py_bfs(g, source, slot)

print(f"distance from {source} to {dest} is {py_distance(g, dest, slot)}")
