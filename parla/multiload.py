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
                self.dlopen("libpython3.7m_parla_stub.so")
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
    wrap_imports: bool

    def __init__(self):
        self.wrap_imports = False
        self.context_stack = [MultiloadContext(0)]

    @property
    def current_context(self):
        return self.context_stack[-1]

    @property
    def in_progress(self) -> list:
        if not hasattr(self, "_in_progress"):
            self._in_progress = dict()
        return self._in_progress

multiload_thread_locals = MultiloadThreadLocals()

# Some convenience functions to ease managing in-progress multiloads

# Mark a multiload as in-progress.
def mark_in_progress(full_name):
    assert full_name not in multiload_thread_locals.in_progress
    multiload_thread_locals.in_progress[full_name] = empty_forwarding_module()

# Check if a multiload for a given name is in-progress.
def check_in_progress(full_name):
    return full_name in multiload_thread_locals.in_progress

def get_in_progress(full_name):
    return multiload_thread_locals.in_progress.get(full_name)

def pop_in_progress(full_name):
    return multiload_thread_locals.in_progress.pop(full_name)

# Create all the replicas/contexts we want
multiload_contexts = [MultiloadContext() if i else MultiloadContext(0) for i in range(NUMBER_OF_REPLICAS)]
for i in range(NUMBER_OF_REPLICAS):
    assert multiload_contexts[i].nsid == i
free_multiload_contexts = list(multiload_contexts)

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

# TODO: This doesn't work quite right yet.
# Modules that are picked up because they are in progress elsewhere should
# return a cached in-progress multiload for their import instead of
# returning the in-progress module returned from the default import.
# Here's an outline of how this should work:
# Every non-in-progress multiload registers an empty forwarding module *before* running the first import.
# Imports that would return an in-progress module now must return the cached in-progress forwarding module.
# The first time an in-progress module is being returned, the import wrapper should register the module object it
# got from the default import into the cached in-progress multiload and insert the in-progress multiload into sys.modules.
# The in-progress module caches should be pulled from a dictionary stored here with behavior similar to sys.modules.

# Correct handling of an in-progress module:
# A multiload object can never be in progress!
# - A multiload on __enter__ creates an entry in the cache of in-progress multiloads.
# - On exit it pops that value out of the in-progress cache and inserts it into sys.modules
#   if the wrapper module object hasn't already been patched in.
# Notes about lazy fromlist loading:
# - Fromlist entries have to be pre-emptively marked as
#   in-progress before we know which ones are modules since
#   this is knowledge we only get after the first import in
#   the multiload runs.
# - In the case of a * import, the * is the only entry in the
#   fromlist. Currently submodules are only included in the *
#   import if they are added as an attribute of their parent
#   module by an import that happens when initializing that module
#   this means that we can just ignore * imports in the fromlist
#   and let the import statement at the parent module level
#   handle the multiload.
# - When sorting through which fromlist entries actually need
#   submodule objects, we also should delete erroneous in-progress
#   entries for things that aren't actually modules.
# An in-progress parent ModuleImport object on capture should:
# - check if the loaded module is a multiload, if it is not:
# - swap in the corresponding dummy multiload from the in progress cache
# - register the loaded module in whatever the current context is
# - Note: if there is a case where there isn't a corresponding
#   multiload in progress in the cache, this corresponds to
#   the case where a previously loaded module is being reloaded
#   as a multiload. This is where that error gets raised.

# Note: The case where someone wants to multiload a submodule of
#   a module that has already been loaded arguably should be allowed,
#   however it's not obvious how to make that case work since
#   each import command actually can pull in any number of modules
#   and the only way we know which ones are intended to be multiloaded
#   by the given import is by comparing with what hasn't already been
#   imported. Forcing multiloading to start at a module's root is
#   a way to avoid subtle errors where things that are implicitly imported
#   before multiloading end up being unintentionally not multiloaded.

# Note: We have to tell users to multiload things first and then
#   import other stuff they need afterward. Otherwise, things they
#   intend to multiload can be picked up as dependencies before the
#   actual multiload happens.

# Note: Overriding imports after the fact, isn't a route we should
#   take because references to the object produced by the first load
#   can be picked up and stored in other places where they may not
#   be find-able later. Getting the referrers to such a module
#   will likely be insufficient since the C API only requires
#   that references not be leaked.

# Note: "register" can just be rolled into the "__exit__".

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

# Substitute an in-progress multiload into
# sys.modules over the top of a new entry created there.
# This operation has to be done aggressively to ensure
# that, even though a module's import was started without
# it being present in sys.modules, any subsequent access
# into sys.modules will pick up the forwarding module
# instead of the original one.
# This has to be done out-of-line from the "capture"
# operation because a module object first becomes visible
# in sys.modules before its builtin __import__ actually
# returns, but references to an in-progress module can already
# be stored before __import__ returns, as is the case
# with mutually recursive imports.
def update_sysmodules_from_in_progress(full_name):
    sys_cached = sys.modules.get(full_name)
    if sys_cached is None:
        return
    # TODO: Is there a way to make this function only
    # be called once we know that the module is not exempt?
    if is_exempt(full_name, sys_cached):
        return
    incomplete_forward = get_in_progress(full_name)
    if is_forwarding(sys_cached):
        assert incomplete_forward is None or sys_cached is incomplete_forward
        return
    assert incomplete_forward is not None
    current_context_id = multiload_thread_locals.current_context.__index__()
    assert current_context_id not in incomplete_forward._parla_base_modules
    incomplete_forward._parla_base_modules[current_context_id] = sys_cached
    sys.modules[full_name] = incomplete_forward

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

# TODO: is_exempt handling elsewhere needs a refactor if possible.
# This routine is okay, but is there any way to bail earlier in
# the main routine so that multiple calls to this aren't needed?
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

def setattr_on_all(module, attr, val):
    for wrapped in module._parla_base_modules.values():
        setattr(wrapped, attr, val)

def delattr_if_present_on_current(module, attr):
    assert is_forwarding(module)
    # TODO: Once the iteration order is refactored to do whole module imports
    # in one pass instead of doing imports into all contexts for each module
    # this try/except won't be necessary. Currently the issue is that
    # if a child import is running the parent module may not actually be
    # there in the given context yet, which is weird.
    try:
        current = module._parla_base_modules[multiload_thread_locals.current_context.__index__()]
    except KeyError:
        return
    if hasattr(current, attr):
        delattr(current, attr)

# When a call to __import__ is made, our override builds a tree of
# objects to track the difference between what was in sys.modules before
# the builtin __import__ is called and what is there afterward.
# This class is a node in that tree. It is responsible for tracking
# what happens with a specific module. The base class ModuleImport covers
# the case of a module that we know a priori does not need to be
# multiloaded by the current import. The subclass ModuleMultiload
# covers the case of a module that does need to be multiloaded.
# We still have to track non-multiloading modules in each given import
# because:
# - The in-progress cache may need to be updated if this is the first time
#   a newly created entry in sys.modules corresponding to the current module
#   has been created.
# - An error needs to be raised in the case of a user requesting a
#   multiload of a module that has already been imported. This case
#   shows up as something that fails may_need_multiload, so we check here.
# - Attribute updates to the corresponding (already loaded) module may still
#   be needed if a multiload is occuring for one of its submodules.

class ModuleImport:

    def __init__(self, full_name, short_name):
        self.submodules = []
        self.full_name = full_name
        self.short_name = short_name
        self.is_pruned = True
        self.was_in_progress = full_name in sys.modules
        self.module_was_present = False
        self.is_multiload = False

    def add_submodule(self, submodule):
        self.submodules.append(submodule)

    def enter_submodules(self):
        for submodule in self.submodules:
            submodule.__enter__()

    def __enter__(self):
        if self.full_name in sys.modules:
            self.module_was_present = True
        if may_need_multiload(self.full_name):
            mark_in_progress(self.full_name)
        else:
            check_for_bad_multiload(self.full_name)
            # Update sysmodules if this is the first time we're
            # seeing the module being built by the builtin __import__.
            update_sysmodules_from_in_progress(self.full_name)
            self.enter_submodules()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.was_in_progress and check_in_progress(self.full_name):
            pop_in_progress(self.full_name)
        for submodule in self.submodules:
            submodule.__exit__(exc_type, exc_val, exc_tb)

    @property
    def in_progress(self):
        try:
            return self.in_progress_cache
        except AttributeError:
            pass
        if check_in_progress(self.full_name):
            self.in_progress_cache = True
            return True
        self.in_progress_cache = False
        return False

    def tag_for_pruning(self):
        self.is_pruned = False

    def prune_non_module_names(self, loaded_module):
        if self.is_pruned:
            return
        # Get rid of any nodes in self.submodules
        # that actually correspond to things other
        # than modules. This has to be done after
        # an import runs, since which names are bound
        # to modules vs objects in a from-style import
        # isn't known until after the import runs.
        new_submodules = []
        for submodule in self.submodules:
            loaded_submodule = sys.modules.get(submodule.full_name)
            # If the requested name doesn't bind to a submodule,
            # it's not going to be present in sys.modules.
            # This can be the case if it's not a module at all or
            # if it's an actual module, just not a submodule
            # of the given parent (fromlists import objects by name
            # and names can be bound to other modules).
            # If it's already in-progress from an ongoing import
            # there will still be some kind of entry in sys.modules.
            if loaded_submodule is None and submodule.is_multiload:
                # Clear out the unneeded entry from the in-progress cache.
                # for most things this is done in the __exit__
                # method, but we're removing this submodule now
                # so __exit__ won't be called later.
                pop_in_progress(submodule.full_name)
            elif loaded_submodule is not None:
                assert type(loaded_submodule) is types.ModuleType
                assert is_submodule(loaded_submodule, loaded_module)
                # We could prune away in-progress modules here too,
                # but their presence still carries meaningful info
                # about what's happening in the current import,
                # so leave them in for the time being.
                new_submodules.append(submodule)
        self.submodules = new_submodules
        self.is_pruned = True

    def capture_submodules(self, loaded_module):
        did_work = False
        for submodule in self.submodules:
            if submodule.is_multiload:
                delattr_if_present_on_current(loaded_module, submodule.short_name)
                did_work = True
            submodule_did_work = submodule.capture()
            did_work = did_work or submodule_did_work
        return did_work

    # See what changed after the import ran,
    # and restore the state to how it was before
    # so that the import can be safely rerun.
    # Two types of changes have to be managed:
    # - changes in sys.modules
    # - changes in module attribute updates based off of from-style imports
    # The first time this is run for a given import
    # it also checks the imported submodules to see if
    # they are actually modules or not since, in general,
    # we can't have this knowledge until the import has actually run.
    # This is because a from-style import can be used for submodules
    # or for objects within a module.
    def capture(self):
        # We don't know a priori if a module is exempt or not,
        # so this is the earliest we can stop.
        if "." not in self.full_name and is_exempt(self.full_name, sys.modules[self.full_name]):
            return False
        if not self.module_was_present and self.full_name in sys.modules:
            self.is_multiload = True
            update_sysmodules_from_in_progress(self.full_name)
            loaded_module = sys.modules.pop(self.full_name)
            assert loaded_module is get_in_progress(self.full_name)
        else:
            loaded_module = sys.modules[self.full_name]
            if not is_forwarding(loaded_module):
                # In some cases it's possible for an in-progress
                # multiload to not have an entry inserted into
                # sys.modules until after a subsequent import runs.
                # This happens because the module object placed
                # into sys.modules doesn't get created until
                # the code for the module is run, and the __init__.py
                # file for the containing folder is run first.
                # More concretely say a.b imports e from a.c.d
                # using a fromlist style import, and a.c's __init__
                # first imports a.c.f which also imports e from
                # a.c.d using a fromlist style import. In this case,
                # the second import will run before the module
                # object for a.c.d is even in sys.modules.
                # If that's the case, this may be the first time
                # we see the newly created module object.
                # In that case, immediately swap it in here.
                update_sysmodules_from_in_progress(self.full_name)
                loaded_module = sys.modules[self.full_name]
        # Multiloading a submodule of a non-multiloaded parent
        # module isn't currently supported, though it may be
        # possible in general, that case isn't debugged yet.
        # We've already returned here in the case of an exempt
        # module, so this has to be true.
        assert is_forwarding(loaded_module)
        # TODO: refactor this out into a separate routine that can be
        # reused in the multiload and non-multiload cases.
        # This is the earliest we can tell if names listed in
        # a fromlist in a from-style import are actually associated
        # with submodules that we need to manage, so prune away
        # other stuff now.
        self.prune_non_module_names(loaded_module)
        return self.capture_submodules(loaded_module) or self.is_multiload

    def register(self):
        if self.is_multiload:
            assert sys.modules.get(self.full_name) is None
            sys.modules[self.full_name] = get_in_progress(self.full_name)
        for submodule in self.submodules:
            submodule.register()
            if submodule.is_multiload:
                loaded_submodule = get_in_progress(submodule.full_name)
                setattr_on_all(loaded_submodule, submodule.short_name, loaded_submodule)

# A module needs to be multiloaded in the current import
# if it hasn't already been imported and if there isn't
# already a multiload in progress.
def may_need_multiload(full_name):
    return full_name not in sys.modules and not check_in_progress(full_name)

# Raise an error if this is a module that ought to be multiloaded,
# but has instead been already imported.
def check_for_bad_multiload(full_name):
    if check_in_progress(full_name):
        return
    module = sys.modules.get(full_name)
    if module is None or is_forwarding(module) or is_exempt(full_name, module):
        return
    raise ImportError("Attempting to multiload module {} which has already been imported normally.".format(full_name))

# Build a tree of ModuleImport or ModuleMultiload objects
# to observe the changes made by the call to the builtin
# __import__ routine. For example, if someone does
# from m1.m2.m3 import n1, n2, n3 the tree has
# m1 as its root. m2 is a child of m1. m3 is a child of m2.
# n1, n2, and n3 are children of m3.
# Fan out only occurs at the lowest level, and only with
# "from" style imports. When a fromlist is nonempty,
# entries in the tree may be created for things that are
# not actually modules since it's not clear what is
# a module or not until after the import runs.
# The non-module entries are trimmed away when the
# "capture" method of the parent module is first called. 
def build_import_tree(full_name, fromlist):
    short_names = full_name.split(".")
    full_names = [".".join(short_names[:i+1]) for i in range(len(short_names))]
    started_multiloading = False
    root = ModuleImport(full_names[0], short_names[0])
    previous = root
    for full_name, short_name in zip(islice(full_names, 1, None), islice(short_names, 1, None)):
        submodule = ModuleImport(full_name, short_name)
        previous.add_submodule(submodule)
        previous = submodule
    if fromlist:
        previous.tag_for_pruning()
        for name in fromlist:
            if name == "*":
                # * imports don't actually trigger imports of submodules.
                # All they do is pick up attributes of the parent module
                # once it has been initialized.
                # Because of this, * imports don't trigger any multiloads.
                continue
            submodule_full_name = ".".join([full_name, name])
            # Some of these names aren't ultimately bound to modules,
            # but we don't know that a priori, so speculatively
            # create tree nodes for all of them.
            submodule = ModuleImport(submodule_full_name, name)
            previous.add_submodule(submodule)
    return root

# The actual function that's used to override __import__.
# It builds a tree of ModuleImport objects
# that represents the import by calling in to build_import_tree.
# It then uses that tree to observe the changes made by running
# the builtin __import__ and capture the results into the wrapper
# module objects with the context-switching-aware __getattr__.
def import_override(name, glob = None, loc = None, fromlist = tuple(), level = 0):
    if multiload_thread_locals.wrap_imports:
        full_name = get_full_name(name, glob, loc, fromlist, level)
        import_tree = build_import_tree(full_name, fromlist)
        with import_tree:
            for context in multiload_contexts:
                with context:
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
