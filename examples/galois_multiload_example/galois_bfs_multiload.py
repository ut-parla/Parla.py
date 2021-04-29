#from threading import Thread

from parla.multiload import multiload, multiload_contexts, mark_module_as_global

VECS = 2


if __name__ == '__main__':
    
    with multiload():
        from parla_galois.core import py_bfs, py_distance
    
    from parla_galois.core import py_load_file, py_init_galois
    py_init_galois()
    g = py_load_file("rmat15.gr")
    

    for i in range(VECS):
        with multiload_contexts[i]:
            py_bfs(g, i*10, i)
            print(py_distance(g, 5, i))
