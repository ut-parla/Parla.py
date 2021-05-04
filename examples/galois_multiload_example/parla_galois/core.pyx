from core cimport *

cdef class PyGraph:
    cdef Graph* underlying

def py_distance(unsigned long self, int node, int slot):
    return read_distance(<Graph*>self, node, slot)

def py_bfs(unsigned long self, int source, int slot):
    return bfs(<Graph*>self, source, slot)


def py_load_file(str fn):
    #r = PyGraph()
    r = <unsigned long>load_file(bytes(fn, "utf-8"))
    return r

def py_init_galois():
    init_galois()

def py_delete_galois():
    delete_galois()

def set_active_threads(int i):
    return setActiveThreads(i)

