"""

.. testsetup::

    import parla
"""

import os
import sys
import threading
import types
import importlib
import builtins
import ctypes
from heapq import merge
from contextlib import contextmanager
from typing import Collection, Optional
from itertools import islice

__all__ = ["multiload", "multiload_context", "MultiloadContext"]

NUMBER_OF_REPLICAS = 10
MAX_REPLICA_ID = 16


# Supervisor wrappers

_parla_supervisor = ctypes.CDLL("libparla_supervisor.so")

context_new = _parla_supervisor.context_new
context_new.argtypes = []
context_new.restype = ctypes.c_long

context_setenv = _parla_supervisor.context_setenv
context_setenv.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]

context_unsetenv = _parla_supervisor.context_unsetenv
context_setenv.argtypes = [ctypes.c_long, ctypes.c_char_p]

context_affinity_override_set_allowed_cpus_py = _parla_supervisor.context_affinity_override_set_allowed_cpus_py
context_affinity_override_set_allowed_cpus_py.argtypes = [ctypes.c_long, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]

context_dlopen = _parla_supervisor.context_dlopen
context_dlopen.argtypes = [ctypes.c_long, ctypes.c_char_p]


class virt_dlopen_state(ctypes.Structure):
    _fields_ = [("enabled", ctypes.c_char),
                ("lm", ctypes.c_long)]

virt_dlopen_swap_state = _parla_supervisor.virt_dlopen_swap_state
virt_dlopen_swap_state.argtypes = [ctypes.c_char, ctypes.c_long]
virt_dlopen_swap_state.restype = virt_dlopen_state


# Context representation


class MultiloadContext():
    nsid: int

    def __init__(self, nsid = None):
        self._thread_local = threading.local()
        if nsid is None:
            nsid = context_new()
            self.nsid = nsid
            # TODO: This name needs to be computed based on the Python version and config, but I have no idea what the "m" is so I'm not going to do that yet.
            self.dlopen("libpython3.7m_parla_stub.so")
        else:
            self.nsid = nsid

    def dispose(self):
        # TODO: Implement unloading of contexts
        raise NotImplementedError()

    def __index__(self):
        return self.nsid

    # Context control API

    def setenv(self, name: str, value):
        context_setenv(self.nsid, name.encode("ascii"), str(value).encode("ascii"), 1)

    def unsetenv(self, name: str):
        context_unsetenv(self.nsid, name.encode("ascii"))

    def set_allowed_cpus(self, cpus: Collection[int]):
        cpu_array = (ctypes.c_int * len(cpus))()
        for i, cpu in enumerate(cpus):
            cpu_array[i] = cpu
        context_affinity_override_set_allowed_cpus_py(self.nsid, len(cpus), cpu_array)

    def dlopen(self, name: str):
        context_dlopen(self.nsid, name.encode("ascii"))

    def force_dlopen_in_context(self):
        virt_dlopen_swap_state(True, self.nsid)

@contextmanager
def run_in_context(context):
    old_context = multiload_thread_locals.current_context
    multiload_thread_locals.current_context = context
    context.force_dlopen_in_context()
    yield
    old_context.force_dlopen_in_context()
    multiload_thread_locals.current_context = old_context

# Create all the replicas/contexts we want
multiload_contexts = [MultiloadContext() if i else MultiloadContext(0) for i in range(NUMBER_OF_REPLICAS)]
for i in range(NUMBER_OF_REPLICAS):
    assert multiload_contexts[i].nsid == i

# Thread local storage wrappers

class MultiloadThreadLocals(threading.local):
    current_context: MultiloadContext
    wrap_imports: bool

    def __init__(self):
        # TODO: This doesn't work like it appears to.
        # "setdefault" doesn't set a default value for thread local storage in all threads.
        # All this does is set the value in the current thread.
        self.__dict__.setdefault("current_context", MultiloadContext(0))
        self.__dict__.setdefault("wrap_imports", False)

    @property
    def in_progress(self) -> list:
        if not hasattr(self, "_in_progress"):
            self._in_progress = []
        return self._in_progress

multiload_thread_locals = MultiloadThreadLocals()

def forward_getattribute(self, attr):
    if attr[:6] == "_parla":
        return object.__getattribute__(self, attr)
    return getattr(object.__getattribute__(self, "_parla_base_modules")[multiload_thread_locals.current_context.__index__()], attr)

def forward_setattr(self, name, value):
    if name[:6] == "_parla":
        return object.__setattr__(self, name, value)
    return object.__getattribute__(self, "_parla_base_modules")[multiload_thread_locals.current_context.__index__()].__setattr__(name, value)

builtin_module_setattr = types.ModuleType.__setattr__

def module_setattr(self, name, value):
    if is_forwarding(self):
        return forward_setattr(self, name, value)
    return builtin_module_setattr(self, name, value)

# Need forbiddenfruit module to do this.
#types.ModuleType.__setattr__ = module_setattr

# __dir__ for modules doesn't actually accept a "self" parameter,
# so instead we need to create a function object that captures the needed info
# on a per-module basis and use that to override __dir__.
# See https://www.python.org/dev/peps/pep-0562/ for details.
def get_forward_dir(module):
    def forward_dir():
        forwarding_dir = object.__dir__(module)
        module_dir = object.__getattribute__(module, "_parla_base_modules")[multiload_thread_locals.current_context.__index__()].__dir__()
        return list(merge(forwarding_dir, module_dir))
    return forward_dir

# Just defining __getattribute__ at module scope isn't actually
# doing the correct thing right now. It's probably a bug.
# Given that though, we can override __getattr__ instead
# and then delete attributes that would normally be found.
# That'll probably use a slower code path, but
# it will work around the bug.
# The signature isn't the same as the usual __getattr__ though,
# as is the case with __dir__. 
# Again see https://www.python.org/dev/peps/pep-0562/ for details.
def get_forward_getattr(module):
    def forward_getattr(name: str):
        return getattr(module._parla_base_modules[multiload_thread_locals.current_context.__index__()], name)
    return forward_getattr

# TODO: Switch this to operating on dictionaries with integer keys instead of lists.
def forward_module(base_modules):
    assert isinstance(base_modules, dict)
    for i in range(NUMBER_OF_REPLICAS):
        assert i in base_modules
    for i in base_modules:
        for j in base_modules:
            if i < j:
                assert base_modules[i] is not base_modules[j]
    forwarding_module = types.ModuleType("")
    # Make absolutely sure every attribute access is forwarded
    # through our __getattr__ by getting rid of the things that
    # would normally be there on a module.
    for name in dir(forwarding_module):
        delattr(forwarding_module, name)
    forwarding_module._parla_forwarding_module = True
    forwarding_module._parla_base_modules = base_modules
    # Overriding __getattribute__ at a module level
    # doesn't actually work right now.
    #forwarding_module.__getattribute__ = forward_getattribute
    forwarding_module.__dir__ = get_forward_dir(forwarding_module)
    # If overriding __getattribute__ ever starts working, this may not be needed.
    forwarding_module.__getattr__ = get_forward_getattr(forwarding_module)
    return forwarding_module

def is_forwarding(module):
    return getattr(module, "_parla_forwarding_module", False)

# Technique to check if something is in the standard library.
# Based loosely off of https://stackoverflow.com/a/22196023.
exempt_cache = set(sys.builtin_module_names)
external_cache = set()
stdlib_base_paths = [os.path.abspath(p) for p in sys.path if p.startswith(sys.prefix) and "site-packages" not in p and p != sys.prefix]

def is_exempt(name, module):
    if name in exempt_cache:
        return True
    if name in external_cache:
        return False
    if not hasattr(module, "__file__"):
        exempt_cache.add(name)
        return True
    fname = module.__file__
    # TODO: can we use something less brittle here?
    if "site-packages" in fname:
        external_cache.add(fname)
        return False
    if any(os.path.abspath(fname).startswith(prefix) for prefix in stdlib_base_paths):
        exempt_cache.add(name)
        return True
    external_cache.add(name)
    return False

builtin_import = builtins.__import__

# TODO: Double check function and usage against
# https://github.com/python/cpython/blob/ffd9753a944916ced659b2c77aebe66a6c9fbab5/Python/import.c#L1843
def get_full_name(name, globals=None, locals=None, fromlist=tuple(), level=0):
    if not level:
        return name
    # Get the full name of something imported using a relative import.
    if globals["__file__"][-11:] == "__init__.py":
        if level > 1:
            root = ".".join(globals["__name__"].split(".")[:-(level-1)])
        else:
            root = globals["__name__"]
    else:
        root = ".".join(globals["__name__"].split(".")[:-level])
    if name:
        return ".".join([root, name])
    return root

def is_submodule(inner, outer):
    return inner.__name__.startswith(outer.__name__)

def deep_delattr_if_present(module, attr):
    try:
        modules_to_modify = module._parla_base_modules
    except AttributeError:
        modules_to_modify = {0 : module}
    for module in modules_to_modify.values():
        try:
            delattr(module, attr)
        except AttributeError:
            pass

def deep_setattr(module, attr, val):
    try:
        modules_to_modify = module._parla_base_modules
    except AttributeError:
        modules_to_modify = {0 : module}
    for module in modules_to_modify.values():
        setattr(module, attr, val)

class ModuleImport:

    def __init__(self, full_name, short_name):
        self.submodules = []
        self.full_name = full_name
        self.short_name = short_name
        self.captured_attrs = []
        self.fromlist = None
        self.fromlist_is_registered = False
        self.loaded_submodules = None
        self.needs_update = None

    def set_fromlist(self, fromlist):
        self.fromlist = fromlist
        # Multiloading of submodules picked up via * imports is deferred until those
        # submodules are imported within the parent module.
        if fromlist and fromlist[0] == "*":
            self.loaded_submodules = []
        else:
            self.loaded_submodules = [item_name for item_name in fromlist if ".".join([self.full_name, item_name]) in sys.modules]

    def add_submodule(self, submodule):
        self.submodules.append(submodule)

    def __enter__(self):
        for submodule in self.submodules:
            submodule.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for submodule in self.submodules:
            submodule.__exit__(exc_type, exc_val, exc_tb)

    @property
    def in_progress(self):
        try:
            return self.in_progress_cache
        except AttributeError:
            pass
        if self.full_name in multiload_thread_locals.in_progress:
            return True

    def register_fromlist(self):
        if self.fromlist_is_registered:
            return
        module = sys.modules[self.full_name]
        submodules_all_forwarding_or_in_progress = True
        for submodule_name in self.fromlist:
            submodule_full_name = ".".join([self.full_name, submodule_name])
            try:
                submodule = getattr(module, submodule_name)
            except AttributeError as a:
                submodule = sys.modules.get(submodule_full_name)
                deep_setattr(module, submodule_name, submodule)
                if submodule is None:
                    raise ImportError("Module {} has no submodule or attribute {}.".format(self.full_name, submodule_name)) from a
            if not isinstance(submodule, types.ModuleType):
                continue
            if not is_submodule(submodule, module):
                continue
            submodule_is_forwarding = is_forwarding(submodule)
            submodule_in_progress = submodule_full_name in multiload_thread_locals.in_progress
            submodules_all_forwarding_or_in_progress = submodules_all_forwarding_or_in_progress and (submodule_is_forwarding or submodule_in_progress)
            if submodule_name in self.loaded_submodules and not submodule_is_forwarding and not submodule_in_progress:
                raise ImportError("Attempting to multiload module {} which was previously imported without multiloading.".format(".".join([full_name, item_name])))
            if not submodule_is_forwarding and not submodule_in_progress:
                self.submodules.append(ModuleMultiload(submodule_full_name, submodule_name))
        self.fromlist_is_registered = True

    def clear_submodule_attrs(self):
        for submodule in submodules:
            if submodule.is_multiload:
                deep_delattr_if_present(sys.modules[self.full_name], submodule.short_name)

    def capture(self):
        if self.fromlist and not self.fromlist_is_registered:
            self.register_fromlist()
        if "." not in self.full_name and self.is_exempt:
            return False
        did_work = False
        for submodule in self.submodules:
            if submodule.is_multiload:
                deep_delattr_if_present(sys.modules[self.full_name], submodule.short_name)
                did_work = True
            submodule_did_work = submodule.capture()
            did_work = did_work or submodule_did_work
        return did_work

    def register(self):
        for submodule in self.submodules:
            submodule.register()
            if submodule.is_multiload:
                loaded_submodule = sys.modules[submodule.full_name]
                if not is_forwarding(loaded_submodule):
                    assert False
                assert is_forwarding(loaded_submodule)
                deep_setattr(sys.modules[self.full_name], submodule.short_name, loaded_submodule)

    @property
    def is_multiload(self):
        return False

    @property
    def is_exempt(self):
        return is_exempt(self, sys.modules[self.full_name])

class ModuleMultiload(ModuleImport):

    def __init__(self, full_name, short_name):
        super().__init__(full_name, short_name)
        self.captured_modules = dict()

    def __enter__(self):
        multiload_thread_locals.in_progress.append(self.full_name)
        if self.fromlist:
            for item_name in self.fromlist:
                multiload_thread_locals.in_progress.append(".".join([self.full_name, item_name]))
        super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        if self.fromlist:
            for item_name in reversed(self.fromlist):
                found_entry = multiload_thread_locals.in_progress.pop()
                assert found_entry == ".".join([self.full_name, item_name])
        found_entry = multiload_thread_locals.in_progress.pop()
        assert found_entry == self.full_name

    def capture(self):
        if self.fromlist and not self.fromlist_is_registered:
            self.register_fromlist()
        if "." not in self.full_name and self.is_exempt:
            return False
        if not self.in_progress:
            raise ImportError("Attempting to multiload module {} which was previously imported without multiloading.".format(self.full_name))
        captured_module = sys.modules.pop(self.full_name)
        assert not hasattr(captured_module, "_parla_context")
        captured_module._parla_context = multiload_thread_locals.current_context
        self.captured_modules[multiload_thread_locals.current_context.__index__()] = captured_module
        for submodule in self.submodules:
            delattr(captured_module, submodule.short_name)
        for submodule in self.submodules:
            submodule.capture()
        return True

    def register(self):
        forward = forward_module(self.captured_modules)
        sys.modules[self.full_name] = forward
        for submodule in self.submodules:
            # All submodules listed in this import tree
            # should be forwarding if this submodule is a forwarding module.
            assert submodule.is_multiload
            loaded_submodule = sys.modules[submodule.full_name]
            assert is_forwarding(loaded_submodule)
            deep_setattr(forward, submodule.short_name, loaded_submodule)

    @property
    def is_multiload(self):
        return True

def may_need_multiload(full_name):
    return full_name not in sys.modules

def build_import_tree(full_name, fromlist):
    short_names = full_name.split(".")
    full_names = [".".join(short_names[:i+1]) for i in range(len(short_names))]
    started_multiloading = False
    if may_need_multiload(full_names[0]):
        root = ModuleMultiload(full_names[0], short_names[0])
        started_multiloading = True
    else:
        root = ModuleImport(full_names[0], short_names[0])
    previous = root
    for full_name, short_name in zip(islice(full_names, 1,None), islice(short_names, 1, None)):
        if may_need_multiload(full_name):
            started_multiloading = True
            submodule = ModuleMultiload(full_name, short_name)
        else:
            assert not started_multiloading
            submodule = ModuleImport(full_name, short_name)
        previous.add_submodule(submodule)
        previous = submodule
    if fromlist:
        previous.set_fromlist(fromlist)
    return root

def import_override(name, glob = None, loc = None, fromlist = tuple(), level = 0):
    if multiload_thread_locals.wrap_imports:
        full_name = get_full_name(name, glob, loc, fromlist, level)
        import_tree = build_import_tree(full_name, fromlist)
        with import_tree:
            for context in multiload_contexts:
                with run_in_context(context):
                    ret = builtin_import(name, glob, loc, fromlist, level)
                    did_work = import_tree.capture()
                    if not did_work:
                        return ret
                assert did_work
            import_tree.register()
        if fromlist:
            return sys.modules[full_name]
        return sys.modules[full_name.split(".", 1)[0]]
    else:
        return builtin_import(name, glob, loc, fromlist, level)
    assert(False)

builtins.__import__ = import_override

@contextmanager
def multiload():
    """
    Define a block of imports that will be instanced for each context.

    >>> with parla.multiload.multiload():
    >>>     import numpy

    """
    assert not multiload_thread_locals.wrap_imports
    multiload_thread_locals.wrap_imports = True
    try:
        yield
    finally:
        multiload_thread_locals.wrap_imports = False

def multiload_context(i):
    return multiload_contexts[i]
