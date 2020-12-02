from threading import Thread

from parla.multiload import multiload_contexts

if __name__ == '__main__':
    m = 3
    def thing(i):
        multiload_contexts[i].load_stub_library("numa")
        with multiload_contexts[i]:
            import parla_galois.core
            print(i)
            parla_galois.core.py_init_galois()
            parla_galois.core.py_bfs(g, i*10, i)
        print(i, parla_galois.core.py_distance(g, 5, i))

    with multiload_contexts[0]:
        import parla_galois.core

    from parla_galois.core import py_load_file, py_init_galois
    py_init_galois()
    g = py_load_file("rmat15.gr")

    ts = []
    for i in range(1, m):
        print("creating", i)
        # thing(i)
        ts.append(Thread(target=thing, args=(i,)))
    for t in ts: t.start()
    for t in ts: t.join()
