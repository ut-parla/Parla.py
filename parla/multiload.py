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

__all__ = ["multiload", "MultiloadContext"]

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

    def __init__(self, nsid=None):
        self._thread_local = threading.local()
        self._thread_local.__dict__.setdefault("old_context", None)
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

    def __enter__(self):
        """
        Enter a context so that any references to multiloaded modules will use the version of the module loaded into that context.
        :param context_id: The context ID to use.

        >>> with parla.multiload.multiload():
        >>>     import numpy
        >>> with multiload_contexts[1]:
        >>>     numpy.array # Will be the context specific array constructor.

        """
        self._thread_local.old_context = multiload_thread_locals.current_context
        multiload_thread_locals.current_context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        multiload_thread_locals.current_context = self._thread_local.old_context
        self._thread_local.old_context = None

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

    @contextmanager
    def force_dlopen_in_context(self):
        old_state = virt_dlopen_swap_state(True, self.nsid)
        try:
            yield
        finally:
            virt_dlopen_swap_state(old_state.enabled, old_state.lm)

# Create all the replicas/contexts we want
multiload_contexts = [MultiloadContext() if i else MultiloadContext(0) for i in range(NUMBER_OF_REPLICAS)]


# Thread local storage wrappers

class MultiloadThreadLocals(threading.local):
    current_context: Optional[MultiloadContext]
    wrap_imports: bool

    def __init__(self):
        self.__dict__.setdefault("current_context", 0)
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
    return getattr(object.__getattribute__(self, "_parla_base_modules")[multiload_thread_locals.current_context], attr)

def forward_setattr(self, name, value):
    if name[:6] == "_parla":
        return object.__setattr__(self, name, value)
    return object.__getattribute__(self, "_parla_base_modules")[multiload_thread_locals.current_context].__setattr__(name, value)

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
        module_dir = object.__getattribute__(module, "_parla_base_modules")[multiload_thread_locals.current_context].__dir__()
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
        return getattr(module._parla_base_modules[multiload_thread_locals.current_context], name)
    return forward_getattr

def forward_module(base_modules):
    for i in range(len(base_modules)):
        for j in range(i):
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

# Good enough for simple modules.
# More work is needed to make submodules/dependencies work right.
# TODO: 
def multi_import(module_name):
    base_modules = [None] * NUMBER_OF_REPLICAS
    for context in multiload_contexts:
        with context.force_dlopen_in_context():
            base_modules[context] = importlib.import_module(module_name)
            del sys.modules[module_name]
    sys.modules[module_name] = forward_module(base_modules)

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
    if "site-packages" in fname:
        external_cache.add(fname)
        return False
    if any(os.path.abspath(fname).startswith(prefix) for prefix in stdlib_base_paths):
        exempt_cache.add(name)
        return True
    external_cache.add(name)
    return False

builtin_import = builtins.__import__

def multiload_module(full_name, first_copy):
    submodules = []
    context = multiload_thread_locals.current_context
    name_parts = full_name.split(".")
    # TODO: is it ever possible for immediate parents to not be the
    # same module each time?
    immediate_parents = None
    if len(name_parts) > 1:
        immediate_parents = []
        short_name = name_parts[-1]
    for other_context in multiload_contexts:
        with other_context:
            if other_context == context and first_module:
                # TODO: WTF is going on here?
                module = first_copy
            else:
                # Note: locals() is an unused parameter to __import__.
                # So don't actually pass it for performance reasons.
                # This import call is equivalent to absolute importing
                # the desired module.
                module = builtin_import(full_name, globals(), dict(), None, 0)
            parent_module = None
            desired_submodule = module
            for submodule_name in name_parts[1:]:
                parent_module = desired_submodule
                desired_submodule = getattr(desired_submodule, submodule_name)
            assert sys.modules[full_name] is desired_submodule
            del sys.modules[full_name]
            if parent_module is not None:
                delattr(parent_module, short_name)
            assert not hasattr(module, "_parla_load_context")
            module._parla_load_context = i
            base_modules.append(module)
            if immediate_parents is not None:
                immediate_parents.append(parent_module)
        

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

@contextmanager
def multiload_in_progress(full_name, fromlist = None):
    multiload_thread_locals.in_progress.append(full_name)
    if fromlist:
        fromlist_full_names = []
        for item in fromlist:
            if item == "*":
                continue
            item_name = ".".join([full_name, item])
            fromlist_full_names.append(item_name)
            multiload_thread_locals.in_progress.append(item_name)
    try:
        yield
    finally:
        if fromlist:
            for item_name in reversed(fromlist_full_names):
                last = multiload_thread_locals.in_progress.pop()
                assert last == item_name
        last = multiload_thread_locals.in_progress.pop()
        assert last == full_name

def is_forwarding_module(module):
    return getattr(module, "_parla_forwarding_module", False)

def is_submodule(inner, outer):
    return inner.__name__.startswith(outer.__name__)

def import_override(name, glob=None, loc=None, fromlist=None, level=0):
    if multiload_thread_locals.wrap_imports:
        full_name = get_full_name(name, glob, loc, fromlist, level)
        was_loaded = full_name in sys.modules
        if fromlist:
            star_present = "*" in fromlist
            if star_present:
                loaded_submodules = set()
                # TODO: This could be more efficient.
                # Currently this will make star imports significantly slower.
                # Speeding it up will likely require some sort of caching scheme though.
                for module_name in sys.modules:
                    if len(module_name) > len(full_name) and module_name.startswith(full_name):
                        loaded_submodules.append(module_name[len(full_name):])
            else:
                loaded_submodules = {item_name for item_name in fromlist if ".".join([full_name, item_name]) in sys.modules}
        with multiload_in_progress(full_name, fromlist):
            returned_module = builtin_import(name, glob, loc, fromlist, level)
        if is_exempt(full_name, returned_module):
            return returned_module
        main_in_progress = full_name in multiload_thread_locals.in_progress
        desired_module = returned_module
        if fromlist:
            fromlist_submodules = []
            fromlist_submodule_names = []
            submodules_needing_multiload = []
            submodules_all_forwarding_or_in_progress = True
            if star_present:
                accessed_items = [item for item in dir(desired_module) if item != "__builtins__"]
                #accessed_items = desired_module.__all__
            else:
                accessed_items = fromlist
            for item_name in accessed_items:
                submodule = getattr(desired_module, item_name)
                if not isinstance(submodule, types.ModuleType):
                    continue
                if not is_submodule(submodule, desired_module):
                    continue
                submodule_is_forwarding = is_forwarding_module(submodule)
                submodule_full_name = ".".join([full_name, item_name])
                submodule_in_progress = submodule_full_name in multiload_thread_locals.in_progress
                submodules_all_forwarding_or_in_progress = submodules_all_forwarding_or_in_progress and (submodule_is_forwarding or submodule_in_progress)
                if item_name in loaded_submodules and not submodule_is_forwarding and not submodule_in_progress:
                    raise ImportError("Attempting to multiload module {} which was previously imported without multiloading.".format(".".join([full_name, item_name])))
                fromlist_submodules.append(submodule)
                fromlist_submodule_names.append(item_name)
                if not submodule_is_forwarding and not submodule_in_progress:
                    submodules_needing_multiload.append(item_name)
            is_forwarding = is_forwarding_module(desired_module)
            main_needs_multiload = not is_forwarding and not main_in_progress
            if submodules_all_forwarding_or_in_progress and not main_needs_multiload:
                return returned_module
            if was_loaded and not is_forwarding and not main_in_progress:
                raise ImportError("Attempting to multiload module {} which was previously imported without multiloading.".format(full_name))
            if main_needs_multiload:
                multiloads = [None] * NUMBER_OF_REPLICAS
        else:
            parts = full_name.split(".")
            parent_module = None
            for submodule_name in parts[1:]:
                parent_module = desired_module
                desired_module = getattr(desired_module, submodule_name)
            is_forwarding = is_forwarding_module(desired_module)
            if was_loaded and not is_forwarding and not main_in_progress:
                raise ImportError("Attempting to multiload module {} which was previously imported without multiloading.".format(full_name))
            if is_forwarding or main_in_progress:
                return returned_module
            end_name = name.split(".")[-1]
            multiloads = [None] * NUMBER_OF_REPLICAS
        outer_context = multiload_thread_locals.current_context
        if fromlist:
            submodule_multiloads = {submodule_name : [] for submodule_name in submodules_needing_multiload}
            submodule_full_names = [".".join([full_name, submodule_name]) for submodule_name in submodules_needing_multiload]
            initial_submodules = [getattr(returned_module, submodule_name) for submodule_name in submodules_needing_multiload]
            # Needs to worry about simultaneously multiloading multiple things.
            with multiload_in_progress(full_name, fromlist):
                # First delete the sys.modules references to any modules we're multiloading.
                # These references may be there from when the module was loaded earlier
                # in this function.
                if main_needs_multiload:
                    del sys.modules[full_name]
                    # TODO: Why is this necessary?
                    # We never return multiloaded modules but sometimes they are found
                    # as attributes of their parent modules later anyway.
                    needs_parent_update = "." in full_name
                    if needs_parent_update:
                        parts = full_name.split(".")
                        end_name = parts[-1]
                        parent_module = sys.modules[".".join(parts[:-1])]
                        delattr(parent_module, end_name)
                for submodule_name in submodules_needing_multiload:
                    delattr(returned_module, submodule_name)
                for submodule_full_name in submodule_full_names:
                    del sys.modules[submodule_full_name]
                importlib.invalidate_caches()
                for inner_context in multiload_contexts:
                    if inner_context == outer_context:
                        if main_needs_multiload:
                            assert not hasattr(returned_module, "_parla_load_context")
                            returned_module._parla_load_context = inner_context
                            multiloads[inner_context] = returned_module
                        for submodule_name, loaded_submodule in zip(submodules_needing_multiload, initial_submodules):
                            assert not hasattr(loaded_submodule, "_parla_load_context")
                            loaded_submodule._parla_load_context = inner_context
                            submodule_multiloads[submodule_name].append(loaded_submodule)
                        continue
                    with inner_context:
                        new_load = builtin_import(name, glob, loc, fromlist, level)
                        if main_needs_multiload:
                            assert not hasattr(new_load, "_parla_load_context")
                            new_load._parla_load_context = inner_context
                            multiloads[inner_context] = new_load
                        for submodule_name, loads in submodule_multiloads.items():
                            new_submodule = getattr(new_load, submodule_name)
                            assert not hasattr(new_submodule, "_parla_load_context")
                            new_submodule._parla_load_context = inner_context
                            loads.append(new_submodule)
                        if main_needs_multiload:
                            del sys.modules[full_name]
                            if needs_parent_update:
                                delattr(parent_module, end_name)
                        for submodule_name in submodules_needing_multiload:
                            delattr(returned_module, submodule_name)
                        for submodule_full_name in submodule_full_names:
                            del sys.modules[submodule_full_name]
                        importlib.invalidate_caches()
            if main_needs_multiload:
                forward = forward_module(multiloads)
                sys.modules[full_name] = forward
                if needs_parent_update:
                    # TODO: Is this loop actually needed? It doesn't use it's index.
                    for context in multiload_contexts:
                        setattr(parent_module, end_name, forward)
            else:
                forward = returned_module
                assert sys.modules[full_name] is forward
            for submodule_name, loads in submodule_multiloads.items():
                submodule_forward = forward_module(loads)
                sys.modules[".".join([full_name, submodule_name])] = submodule_forward
                if main_needs_multiload or is_forwarding:
                    for context in multiload_contexts:
                        with context:
                            setattr(forward, submodule_name, submodule_forward)
                else:
                    setattr(forward, submodule_name, submodule_forward)
            return forward
        else:
            # Needs to worry about updating parent module attribute.
            if parent_module:
                delattr(parent_module, end_name)
            previous = sys.modules[full_name]
            del sys.modules[full_name]
            importlib.invalidate_caches()
            for inner_context in multiload_contexts:
                if inner_context == outer_context:
                    assert not hasattr(desired_module, "_parla_load_context")
                    desired_module._parla_load_context = inner_context
                    multiloads[inner_context] = desired_module
                    continue
                with inner_context:
                    new_load = builtin_import(name, glob, loc, fromlist, level)
                    found_parent = None
                    new_desired_module = new_load
                    for submodule_name in parts[1:]:
                        found_parent = new_desired_module
                        new_desired_module = getattr(new_desired_module, submodule_name)
                    assert found_parent is parent_module
                    assert not hasattr(new_desired_module, "_parla_load_context")
                    new_desired_module._parla_load_context = inner_context
                    multiloads[inner_context] = new_desired_module
                    if parent_module:
                        delattr(parent_module, end_name)
                    del sys.modules[full_name]
                    importlib.invalidate_caches()
            # Now build and register the forwarding module.
            forward = forward_module(multiloads)
            sys.modules[full_name] = forward
            if parent_module:
                setattr(parent_module, end_name, forward)
                return returned_module
            return forward
        # TODO: Switch iterating over a dict to just a list of lists to preserve orderings.
        # This will allow simultaneous iteration in some places
        # (like when registering the forwarding modules).
    return builtin_import(name, glob, loc, fromlist, level)

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
