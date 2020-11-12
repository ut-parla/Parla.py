"""

.. testsetup::

    import parla
"""

from distutils.sysconfig import get_config_var
import os
import sys
import gc
import threading
import types
import importlib
import builtins
import ctypes
from heapq import merge
from contextlib import contextmanager
from typing import Collection, Optional, Callable, List, Any
from itertools import islice

from .environments import EnvironmentComponentDescriptor, EnvironmentComponentInstance, TaskEnvironment

#from forbiddenfruit import curse

__all__ = ["multiload", "MultiloadContext", "MultiloadComponent", "CPUAffinity"]

NUMBER_OF_REPLICAS = 12
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

# Load a bunch of extension modules to fill the cache

for i in range(128):
    importlib.import_module(f"parla.cache_filler_{i}")

module_spec_cache = None
for item in gc.get_objects():
    if type(item) is dict:
        for key in item:
            if type(key) is tuple and len(key) == 2 and key[1] == "parla.cache_filler_0":
                module_spec_cache = item

assert module_spec_cache is not None
# Our cache of module spec objects that we use
# to update the main module_spec_cache.
module_spec_caches_for_contexts = dict()

# Context representation

class MultiloadContext():
    nsid: int

    def __init__(self, nsid = None):
        if nsid is None:
            nsid = context_new()
            assert nsid >= 0
        self.nsid = nsid
        if nsid:
            # This isn't needed for namespace 0 since the normal libpython is already loaded there.
            # TODO: This name needs to be computed based on the Python version and config, but I have no idea what the "m" is so I'm not going to do that yet.
            with self:
                self.saved_rtld = sys.getdlopenflags()
                sys.setdlopenflags(self.saved_rtld | ctypes.RTLD_GLOBAL)
                self.dlopen("libutil.so.1")
                self.dlopen("librt.so.1")
                self.dlopen("libm.so.6")
                self.dlopen("libpthread.so.0")
                libpython_name_prefix = get_config_var("INSTSONAME").rstrip(".a").split(".so")[0]
                self.dlopen("{}_parla_stub.so".format(libpython_name_prefix))
                sys.setdlopenflags(self.saved_rtld)

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
        r = context_dlopen(self.nsid, name.encode("ascii"))
        if not r:
            raise RuntimeError("Failed to load " + name)

    def force_dlopen_in_context(self):
        virt_dlopen_swap_state(True, self.nsid)

    def __enter__(self):
        multiload_thread_locals.context_stack.append(self)
        self.force_dlopen_in_context()
        return self

    def __exit__(self, *args):
        removed = multiload_thread_locals.context_stack.pop()
        assert removed is self
        multiload_thread_locals.context_stack[-1].force_dlopen_in_context()

# Thread local storage wrappers

class MultiloadThreadLocals(threading.local):
    context_stack: List[MultiloadContext]
    multiloading: bool

    def __init__(self):
        self.multiloading = False
        self.context_stack = [MultiloadContext(0)]

    @property
    def current_context(self):
        return self.context_stack[-1]

    @property
    def in_progress(self) -> dict:
        if not hasattr(self, "_in_progress"):
            self._in_progress = dict()
        return self._in_progress

multiload_thread_locals = MultiloadThreadLocals()

# Create all the replicas/contexts we want
multiload_contexts = [MultiloadContext() if i else MultiloadContext(0) for i in range(NUMBER_OF_REPLICAS)]
for i in range(NUMBER_OF_REPLICAS):
    assert multiload_contexts[i].nsid == i
free_multiload_contexts = list(multiload_contexts)

def get_context_module_specs():
    index = multiload_thread_locals.current_context.__index__()
    context_specs = module_spec_caches_for_contexts.get(index)
    if context_specs is None:
        context_specs = dict()
        module_spec_caches_for_contexts[index] = context_specs
    return context_specs

def allocate_multiload_context() -> MultiloadContext:
    return free_multiload_contexts.pop()

def forward_getattribute(self, attr):
    if attr[:6] == "_parla":
        return object.__getattribute__(self, attr)
    return getattr(object.__getattribute__(self, "_parla_base_modules")[multiload_thread_locals.current_context.__index__()], attr)

def forward_setattr(self, name, value):
    if name[:6] == "_parla":
        return object.__setattr__(self, name, value)
    return object.__getattribute__(self, "_parla_base_modules")[multiload_thread_locals.current_context.__index__()].__setattr__(name, value)

# Hopefully we'll overload this later using a technique like
# the one used in forbiddenfruit, so cache the original version first.
builtin_module_setattr = types.ModuleType.__setattr__

def module_setattr(self, name, value):
    if is_forwarding(self):
        return forward_setattr(self, name, value)
    return builtin_module_setattr(self, name, value)

# Need forbiddenfruit module to do this.
#curse(types.ModuleType, "__setattr__", module_setattr)
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
        # Hack around the fact that the frozen importlib
        # accesses __path__ from the parent module while importing
        # submodules.
        # The real fix is to reorganize things so that imports of
        # modules are done into each environment separately.
        # That reorganization is also what's needed to support
        # loading distinct sets of modules into each context.
        if name == "__path__":
            return getattr(module._parla_base_modules[0], name)
        return getattr(module._parla_base_modules[multiload_thread_locals.current_context.__index__()], name)
    return forward_getattr

def empty_forwarding_module():
    forwarding_module = types.ModuleType("")
    for name in dir(forwarding_module):
        delattr(forwarding_module, name)
    forwarding_module._parla_forwarding_module = True
    forwarding_module._parla_base_modules = dict()
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
exempt_cache.add("parla")
external_cache = set()
stdlib_base_paths = [os.path.abspath(p) for p in sys.path if p.startswith(sys.prefix) and "site-packages" not in p and p != sys.prefix]

def is_exempt(name, module):
    if name in exempt_cache:
        return True
    if name in external_cache:
        return False
    if is_forwarding(module):
        # It's already in-progress, so not exempt.
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

def register_module(forwarding_module, module):
    context = multiload_thread_locals.current_context.__index__()
    assert context not in forwarding_module._parla_base_modules
    forwarding_module._parla_base_modules[context] = module

def get_module_for_current_context(module):
    assert is_forwarding(module)
    return module._parla_base_modules.get(multiload_thread_locals.current_context.__index__())
def check_module_for_current_context(module):
    assert is_forwarding(module)
    return multiload_thread_locals.current_context.__index__() in module._parla_base_modules

class ModuleImport:
    def __init__(self, name):
        self.name = name
    def is_submodule_name(self, name):
        prefix = self.name + "."
        return name.startswith(prefix) or name == self.name
    def collect_names_from_sys(self):
        existing_names = []
        prefix = self.name + "."
        for submodule_name in sys.modules.keys():
            if not self.is_submodule_name(submodule_name):
                continue
            existing_names.append(submodule_name)
        return existing_names
    def cache_module_specs(self):
        context_specs = get_context_module_specs()
        for key, spec in module_spec_cache.items():
            path, submodule_name = key
            if self.is_submodule_name(submodule_name):
                updated = module_spec_cache[key]
                current = context_specs.get(key)
                if current is None:
                    context_specs[key] = updated
                else:
                    assert current is updated
    def restore_module_specs(self):
        context_specs = get_context_module_specs()
        present_keys = []
        for key in module_spec_cache:
            path, submodule_name = key
            if self.is_submodule_name(submodule_name):
                present_keys.append(key)
        for key in present_keys:
            module_spec_cache.pop(key)
        for key, spec in context_specs.items():
            module_spec_cache[key] = spec
    def __enter__(self):
        entries = dict()
        multiload_thread_locals.in_progress[self.name] = entries
        existing_names = self.collect_names_from_sys()
        entries.update({name : sys.modules.pop(name) for name in existing_names})
        for submodule_name, submodule in entries.items():
            assert is_forwarding(submodule)
            wrapped_submodule = get_module_for_current_context(submodule)
            if wrapped_submodule is None:
                continue
            sys.modules[submodule_name] = wrapped_submodule
            for attr_name in wrapped_submodule.__dict__.keys():
                attr = getattr(wrapped_submodule, attr_name)
                if type(attr) is types.ModuleType:
                    if self.is_submodule_name(attr.__name__):
                        wrapped_attr = get_module_for_current_context(attr)
                        assert wrapped_attr is not None
                        setattr(wrapped_submodule, attr_name, wrapped_attr)
        self.restore_module_specs()
        importlib.invalidate_caches()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name in sys.modules and is_exempt(self.name, sys.modules[self.name]):
            multiload_thread_locals.in_progress.pop(self.name)
            return
        forwarding_modules = multiload_thread_locals.in_progress[self.name]
        existing_names = self.collect_names_from_sys()
        for submodule_name in existing_names:
            module = forwarding_modules.get(submodule_name)
            if module is None:
                module = empty_forwarding_module()
                forwarding_modules[submodule_name] = module
            current_wrapped = get_module_for_current_context(module)
            if current_wrapped is None:
                module_to_insert = sys.modules[submodule_name]
                register_module(module, module_to_insert)
            assert get_module_for_current_context(module) is sys.modules[submodule_name]
        for forward_name, forward in forwarding_modules.items():
            current = get_module_for_current_context(forward)
            if forward_name in sys.modules:
                assert current is sys.modules[forward_name]
            assert forward is not None and is_forwarding(forward)
            sys.modules[forward_name] = forward
            if current is not None:
                # Use __dict__.keys() since numpy uses a lazy import
                # for numpy.testing and this logic breaks when
                # a lazy import is triggered here.
                for attr_name in current.__dict__.keys():
                    attr = getattr(current, attr_name)
                    if type(attr) is types.ModuleType:
                        if self.is_submodule_name(attr.__name__):
                            setattr(current, attr_name, forwarding_modules[attr.__name__])
        self.cache_module_specs()
        multiload_thread_locals.in_progress.pop(self.name)

# Our modifications to the import machinery aren't
# thread-safe unless concurrent threads are importing completely disjoint modules.
# These locks manage that.
module_import_locks = dict()

def import_in_current(name, glob = None, loc = None, fromlist = tuple(), level = 0):
    full_name = get_full_name(name, glob, loc, fromlist, level)
    base_name = full_name.split(".", 1)[0]
    if base_name not in module_import_locks:
        module_import_locks[base_name] = threading.RLock()
    with module_import_locks[base_name]:
        if base_name in multiload_thread_locals.in_progress:
            builtin_import(name, glob, loc, fromlist, level)
            return
        if base_name in sys.modules:
            current_base = sys.modules[base_name]
            if is_exempt(base_name, current_base):
                builtin_import(name, glob, loc, fromlist, level)
                return
            if not is_forwarding(current_base) and not is_exempt(base_name, current_base):
                raise ImportError("Attempting to import module {} within a given execution context that has already been imported globally".format(base_name))
        with ModuleImport(base_name):
            builtin_import(name, glob, loc, fromlist, level)

@contextmanager
def outermost_multiload_here():
    assert multiload_thread_locals.multiloading
    multiload_thread_locals.multiloading = False
    try:
        yield
    finally:
        multiload_thread_locals.multiloading = True

def import_override(name, glob = None, loc = None, fromlist = tuple(), level = 0):
    if multiload_thread_locals.multiloading:
        with outermost_multiload_here():
            for context in multiload_contexts:
                with context:
                    import_in_current(name, glob, loc, fromlist, level)
    else:
        import_in_current(name, glob, loc, fromlist, level)
    full_name = get_full_name(name, glob, loc, fromlist, level)
    if fromlist:
        return sys.modules[full_name]
    return sys.modules[full_name.split(".", 1)[0]]

builtins.__import__ = import_override

@contextmanager
def multiload():
    """
    Define a block of imports that will be instanced for each context.

    >>> with parla.multiload.multiload():
    >>>     import numpy

    """
    assert not multiload_thread_locals.multiloading
    multiload_thread_locals.multiloading = True
    try:
        yield
    finally:
       multiload_thread_locals.multiloading = False

# Integration with parla.environments

class MultiloadComponentInstance(EnvironmentComponentInstance):
    multiload_context: MultiloadContext

    def __init__(self, descriptor: "MultiloadComponent", env: TaskEnvironment):
        super().__init__(descriptor)
        self.multiload_context = allocate_multiload_context()
        # TODO: Implement loading of specific libraries and appropriate handling of tags. Should tags be used to
        #  trigger library load? Should library loads set tags?
        for setup in descriptor.setup_functions:
            setup(self, env)

    def __enter__(self):
        return self.multiload_context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.multiload_context.__exit__(exc_type, exc_val, exc_tb)

class MultiloadComponent(EnvironmentComponentDescriptor):
    def __init__(self, setup_functions: Collection[Callable[[MultiloadComponentInstance, TaskEnvironment], None]] = ()):
        self.setup_functions = setup_functions

    def combine(self, other):
        assert isinstance(other, MultiloadComponent)
        return MultiloadComponent(tuple(self.setup_functions) + tuple(other.setup_functions))

    def __call__(self, env: TaskEnvironment) -> MultiloadComponentInstance:
        return MultiloadComponentInstance(self, env)

def CPUAffinity(multi: MultiloadComponentInstance, env: TaskEnvironment):
    # Collect CPU numbers and set the affinity
    import parla.cpu
    cpus = []
    for d in env.placement:
        if d.architecture is parla.cpu.cpu:
            cpus.append(d.index)
    multi.multiload_context.set_allowed_cpus(cpus)
