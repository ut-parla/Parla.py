from libc.stdio cimport fprintf, stderr
from libc.stdint cimport intptr_t

cdef extern from "Python.h":
    ctypedef struct PyObject:
        long ob_refcnt
    ctypedef object(*getattrofunc)(object, object)
    ctypedef struct PyTypeObject:
        getattrofunc tp_getattro

from contextlib import contextmanager
from types import ModuleType

from .multiload import address_of_virt_dlopen_get_state

ctypedef struct virt_dlopen_state:
    char enabled
    long int lm

ctypedef virt_dlopen_state(*virt_dlopen_get_state_t)()
cdef virt_dlopen_get_state_t virt_dlopen_get_state = <virt_dlopen_get_state_t><intptr_t>address_of_virt_dlopen_get_state

cdef long int get_current():
    return virt_dlopen_get_state().lm

cdef getattrofunc old_module_getattro = (<PyTypeObject*>ModuleType).tp_getattro

cdef str base_modules_name = "_parla_base_modules"
cdef str dict_name = "__dict__"

def empty_forwarding_module():
    forwarding_module = ModuleType("")
    base_modules = dict()
    forwarding_module._parla_base_modules = base_modules
    return forwarding_module

cpdef dict get_base_modules(module):
    try:
        base_modules = old_module_getattro(module, base_modules_name)
        return base_modules
    except AttributeError:
        return None

def is_forwarding(module):
    ret = get_base_modules(module) is not None
    return ret

cdef int count22 = 0

cdef object new_module_getattro(object module, object attr):
    base_modules = get_base_modules(module)
    if base_modules is None:
        ret = old_module_getattro(module, attr)
        return ret
    (&count22)[0] += 1
    cdef object py_attr = attr.encode()
    cdef char* c_attr = py_attr
    wrapped_module = base_modules[get_current()]
    ret = old_module_getattro(wrapped_module, attr)
    return ret

(<PyTypeObject*>ModuleType).tp_getattro = new_module_getattro
