from threading import Thread, Barrier
from timer import Timer
import sys, os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("env", help="environment to use, must be vec or default", default="default")
parser.add_argument("input", help="Path to input graph")
parser.add_argument("vecs", help="# of threads/vecs to run", type=int)
parser.add_argument("rounds", help="# of rounds to run", type=int)
parser.add_argument('cores', type=int, nargs="*")
args = parser.parse_args()

allowed_ncpus = {}
if args.env == "vec":
    import parla.vec_backtrace
    from parla.multiload import multiload, multiload_contexts, mark_module_as_global

# Set cores for each VEC
vecn = 0
for i in range(0, len(args.cores), 2):
    start = args.cores[i]
    end   = args.cores[i+1] 
    if start == end:
        print(f"Skipping set_allowed_cpus for VEC #{vecn} (this will break stuff, just list the cpu range)")
    else:
        print(f"Setting set_allowed_cpus for VEC #{vecn} to {start}-{end}")
        if args.env == "vec":
            multiload_contexts[vecn].set_allowed_cpus([i for i in range(start, end)])
    allowed_ncpus[vecn] = end-start
    vecn += 1

if args.env == "vec":
    # Init Galois on each VEC
    #if args.env == "vec":
    with multiload():
        import parla_galois.core

# Function each thread will run
def bfs_sssp(i, rounds, skip_rounds, barrier):
    #if args.env == "default":
    import parla_galois.core

    # trick non-vec to read limited cpu
    if args.env == "default":
        os.environ["VECID"] = "0"

    with Timer.get_handle("init-galois"):
        parla_galois.core.py_init_galois(allowed_ncpus[i])
    with Timer.get_handle("load-graph"):
        g = parla_galois.core.py_load_file(args.input)
    
    with Timer.get_handle("all_bfs_executions"):
        done = 0
        for rd in range(rounds+1):
            # sync so everyone runs concurrently
            idx = barrier.wait()
            if idx == 0: barrier.reset()

            if rd < skip_rounds or done == rounds:
                continue
            done += 1

            #source = i
            #report = (i+1)*5
            source = 0
            report = 5
            slot = i
            with Timer.get_handle("every_bfs"):  
                with Timer.get_handle(f"bfs_vec{i}_rd{rd}"):
                    parla_galois.core.py_bfs(g, source, slot)
                    print(f"VEC #{i} round {rd}: distance from {source} to {report} at slot {slot} is {parla_galois.core.py_distance(g, report, slot)}")
    
    parla_galois.core.py_delete_galois()

def vec_bfs_sssp(i, rounds, skip_rounds, barrier):
    with multiload_contexts[i]:
        bfs_sssp(i, rounds, skip_rounds, barrier)

def default_bfs_sssp(i, rounds, skip_rounds, barrier):
    bfs_sssp(i, rounds, skip_rounds, barrier)

barrier = Barrier(args.vecs)
threads = []
if args.env == "vec":
    print("Using VECs as environment")
    thread_fn = vec_bfs_sssp
elif args.env == "default":
    print("Using default environment")
    thread_fn = default_bfs_sssp
else:
    print(f"Invalid env {args.env}")
    sys.exit(0)

for i in range(args.vecs):
    threads.append(Thread(target=thread_fn, args=(i, args.rounds, i, barrier)))

with Timer.get_handle("end-to-end"):
    for t in threads: t.start()
    for t in threads: t.join()

Timer.print()
