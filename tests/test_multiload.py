import types

import pytest


def is_submodule(root, module):
    return module.__name__.startswith(root.__name__ + ".") or module.__name__ == root.__name__

def check_module(mod):
    root_name = mod.__name__
    submodules = {mod}
    def collect_submodules(submodule):
        for name in dir(submodule):
            obj = getattr(submodule, name)
            if type(obj) is types.ModuleType:
                if is_submodule(mod, obj) and obj not in submodules:
                    submodules.add(obj)
                    collect_submodules(obj)
    collect_submodules(mod)
    for submodule in submodules:
        print(submodule.__name__)
    for submodule in submodules:
        assert hasattr(submodule, "_parla_base_modules")
        for key1, module1 in submodule._parla_base_modules.items():
            for key2, module2 in submodule._parla_base_modules.items():
                if key1 != key2:
                    assert module1 is not module2


def test_numpy():
    import ctypes
    import sys
    try:
        from parla.multiload import multiload, multiload_contexts as contexts, multiload_thread_locals
    except OSError as e:
        if "libparla_supervisor.so: cannot open shared object file: No such file or directory" in str(e):
            pytest.xfail(str(e))
        else:
            raise
    assert "numpy" not in sys.modules
    with multiload():
        import numpy
    def check_module(mod):
        root_name = mod.__name__
        submodules = {mod}
        def collect_submodules(submodule):
            for name in dir(submodule):
                obj = getattr(submodule, name)
                if type(obj) is types.ModuleType:
                    if is_submodule(mod, obj) and obj not in submodules:
                        submodules.add(obj)
                        collect_submodules(obj)
        collect_submodules(mod)
        for submodule in submodules:
            print(submodule.__name__)
        for submodule in submodules:
            assert hasattr(submodule, "_parla_base_modules")
            for key1, module1 in submodule._parla_base_modules.items():
                for key2, module2 in submodule._parla_base_modules.items():
                    if key1 != key2:
                        assert module1 is not module2
    check_module(numpy)
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    umath_apis = []
    umath_api_capsules = []
    #print(dir(numpy.core._multiarray_umath._UFUNC_API))
    for context in contexts:
        with context:
            #print(multiload_thread_locals.current_context.nsid)
            umath_apis.append(ctypes.pythonapi.PyCapsule_GetPointer(numpy.core._multiarray_umath._UFUNC_API, None))
            umath_api_capsules.append(id(numpy.core._multiarray_umath._UFUNC_API))
    umath_apis_2 = []
    umath_api_capsules_2 = []
    tanh_copies = []
    for module in numpy.core._multiarray_umath._parla_base_modules.values():
        umath_apis_2.append(ctypes.pythonapi.PyCapsule_GetPointer(module._UFUNC_API, None))
        umath_api_capsules_2.append(id(module._UFUNC_API))
        tanh_copies.append(id(module.tanh))
    assert umath_apis == umath_apis_2
    assert umath_api_capsules == umath_api_capsules_2
    array_apis = []
    for context in contexts:
        with context:
            array_apis.append(ctypes.pythonapi.PyCapsule_GetPointer(numpy.core._multiarray_umath._ARRAY_API, None))
    mt_classes = []
    mt_vtables = []
    mt_classes = []
    rand_ids = []
    for context in contexts:
        with context:
            mt_vtables.append(ctypes.pythonapi.PyCapsule_GetPointer(numpy.random._mt19937.MT19937.__pyx_vtable__, None))
            mt_classes.append(id(numpy.random._mt19937.MT19937))
            rand_ids.append(id(numpy.random.mtrand.rand))
    for i in range(len(contexts)):
        for j in range(len(contexts)):
            if i != j:
                assert mt_vtables[i] != mt_vtables[j]
                assert umath_apis[i] != umath_apis[j]
                assert array_apis[i] != array_apis[j]
                assert tanh_copies[i] != tanh_copies[j]
                assert umath_api_capsules[i] != umath_api_capsules[j]

# from parla.multiload import multiload, run_in_context, multiload_context
#
# def test_multiload():
#     with multiload():
#         import multiload_test_module as mod
#     mod.unused_id = None
#     assert not mod._parla_context.nsid
#     with run_in_context(multiload_context(1)):
#         assert multiload_context(1) == mod._parla_context
#         assert getattr(mod, "_parla_forwarding_module", False)
#         # Need forbiddenfruit to make this last one work.
#         #assert not hasattr(mod, "unused_id")
#
# def test_mutual_recursion():
#     with multiload():
#         import multiload_test_module as mod
#     print(dir(mod.mutual_1))
#     print(dir(mod.mutual_2))
#     assert hasattr(mod.mutual_1, "_parla_forwarding_module")
#     assert hasattr(mod.mutual_2, "_parla_forwarding_module")
#     mod.mutual_2.check()
#     mod.mutual_1.check()
#
# def test_multiple_contexts():
#     import timeit
#     multiload_context(1).set_allowed_cpus([0])
#     multiload_context(2).set_allowed_cpus([1,2,3,4,5,6,7])
#     with multiload():
#         import numpy as np
#     def timed_thing():
#         np.int(1)
#         return 0
#         # return timeit.timeit(lambda: np.dot(np.random.rand(2000, 2000), np.random.rand(2000, 2000)), number=1)
#     with multiload_context(1):
#         ctx_1 = timed_thing()
#     with multiload_context(2):
#         ctx_2 = timed_thing()
#     # assert ctx_2 > ctx_1*1.5
