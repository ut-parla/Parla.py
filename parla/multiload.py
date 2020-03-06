import os
import gc
import sys
import threading
import types
import importlib
import builtins
from heapq import merge
from contextlib import contextmanager

__all__ = ["multiload_context", "multiload"]

num_copies = 10

class threadlocal_default:

    def __init__(self, default):
        self.default = default
        self.val = threading.local()

    def set(self, val):
        self.val.val = val

class threadlocal_default_int(threadlocal_default):

    def __init__(self, default = 0):
        super().__init__(default)

    def __index__(self):
        return getattr(self.val, "val", self.default)

    def __int__(self):
        return self.__index__()

class threadlocal_default_bool(threadlocal_default):

    def __init__(self, default = False):
        super().__init__(default)

    def __bool__(self):
        return getattr(self.val, "val", self.default)

class threadlocal_default_emptylist:

    def __init__(self):
        self.local = threading.local()

    def get(self):
        if hasattr(self.local, "val"):
            return getattr(self.local, "val")
        self.local.val = list()
        return self.local.val

current_context = threadlocal_default_int()

def forward_getattribute(self, attr):
    if attr[:6] == "_parla":
        return object.__getattribute__(self, attr)
    return getattr(object.__getattribute__(self, "_parla_base_modules")[current_context], attr)

def forward_setattr(self, name, value):
    print("setting attribute {}".format(name))
    if name[:6] == "_parla":
        return object.__setattr__(self, name, value)
    return object.__getattribute__(self, "_parla_base_modules")[current_context].__setattr__(name, value)

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
        module_dir = object.__getattribute__(module, "_parla_base_modules")[current_context].__dir__()
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
        return getattr(module._parla_base_modules[current_context], name)
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
    base_modules = []
    for i in range(num_copies):
        base_modules.append(importlib.import_module(module_name))
        del sys.modules[module_name]
    sys.modules[module_name] = forward_module(base_modules)

@contextmanager
def multiload_context(context_id):
    old_context = int(current_context)
    current_context.set(context_id)
    yield
    current_context.set(old_context)

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

wrap_imports = threadlocal_default_bool()

builtin_import = builtins.__import__

def multiload_module(full_name, first_copy):
    submodules = []
    context = int(current_context)
    name_parts = full_name.split(".")
    # TODO: is it ever possible for immediate parents to not be the
    # same module each time?
    immediate_parents = None
    if len(name_parts) > 1:
        immediate_parents = []
        short_name = name_parts[-1]
    for i in range(num_copies):
        with multiload_context(i):
            if i == context and first_module:
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
            assert not hasattr(module, "_parla_load_id")
            module._parla_load_id = i
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

in_progress = threadlocal_default_emptylist()

@contextmanager
def multiload_in_progress(full_name, fromlist = None):
    in_progress.get().append(full_name)
    if fromlist:
        fromlist_full_names = []
        for item in fromlist:
            item_name = ".".join([full_name, item])
            fromlist_full_names.append(item_name)
            in_progress.get().append(item_name)
    yield
    if fromlist:
        for item_name in reversed(fromlist_full_names):
            last = in_progress.get().pop()
            assert last == item_name
    last = in_progress.get().pop()
    assert last == full_name

count = 0

def is_forwarding_module(module):
    return getattr(module, "_parla_forwarding_module", False)

def import_override(name, glob=None, loc=None, fromlist=None, level=0):
    if wrap_imports:
        full_name = get_full_name(name, glob, loc, fromlist, level)
        #with multiload_in_progress(full_name):
        was_loaded = full_name in sys.modules
        if fromlist:
            fromlist_was_loaded = [".".join([full_name, item_name]) in sys.modules for item_name in fromlist]
        with multiload_in_progress(full_name, fromlist):
            returned_module = builtin_import(name, glob, loc, fromlist, level)
        if is_exempt(full_name, returned_module):
            return returned_module
        main_in_progress = full_name in in_progress.get()
        desired_module = returned_module
        if fromlist:
            fromlist_submodules = []
            fromlist_submodule_names = []
            submodules_needing_multiload = []
            submodules_all_forwarding_or_in_progress = True
            for item_name, submodule_was_loaded in zip(fromlist, fromlist_was_loaded):
                submodule = getattr(returned_module, item_name)
                if not isinstance(submodule, types.ModuleType):
                    continue
                submodule_is_forwarding = is_forwarding_module(submodule)
                submodule_full_name = ".".join([full_name, item_name])
                submodule_in_progress = submodule_full_name in in_progress.get()
                submodules_all_forwarding_or_in_progress = submodules_all_forwarding_or_in_progress and (submodule_is_forwarding or submodule_in_progress)
                if submodule_was_loaded and not submodule_is_forwarding and not submodule_in_progress:
                    raise ImportError("Attempting to multiload module {} which was previously imported without multiloading.".format(".".join([full_name, item_name])))
                fromlist_submodules.append(submodule)
                fromlist_submodule_names.append(item_name)
                if not submodule_is_forwarding and not submodule_in_progress:
                    submodules_needing_multiload.append(item_name)
            is_forwarding = is_forwarding_module(desired_module)
            main_needs_multiload = not is_forwarding and not main_in_progress
            if submodules_all_forwarding_or_in_progress and not main_needs_multiload:
                return returned_module
            if main_needs_multiload:
                multiloads = []
        else:
            parts = full_name.split(".")
            parent_module = None
            for submodule_name in parts[1:]:
                parent_module = desired_module
                desired_module = getattr(desired_module, submodule_name)
            is_forwarding = is_forwarding_module(desired_module)
            if is_forwarding or main_in_progress:
                return returned_module
            end_name = name.split(".")[-1]
            multiloads = []
        outer_context_id = int(current_context)
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
                for i in range(num_copies):
                    if i == outer_context_id:
                        if main_needs_multiload:
                            assert not hasattr(returned_module, "_parla_load_id")
                            returned_module._parla_load_id = i
                            multiloads.append(returned_module)
                        for submodule_name, loaded_submodule in zip(submodules_needing_multiload, initial_submodules):
                            assert not hasattr(loaded_submodule, "_parla_load_id")
                            loaded_submodule._parla_load_id = i
                            submodule_multiloads[submodule_name].append(loaded_submodule)
                        continue
                    with multiload_context(i):
                        new_load = builtin_import(name, glob, loc, fromlist, level)
                        if main_needs_multiload:
                            assert not hasattr(new_load, "_parla_load_id")
                            new_load._parla_load_id = i
                            multiloads.append(new_load)
                        for submodule_name, loads in submodule_multiloads.items():
                            new_submodule = getattr(new_load, submodule_name)
                            assert not hasattr(new_submodule, "_parla_load_id")
                            new_submodule._parla_load_id = i
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
                    for i in range(num_copies):
                        setattr(parent_module, end_name, forward)
            else:
                forward = returned_module
                assert sys.modules[full_name] is forward
            for submodule_name, loads in submodule_multiloads.items():
                submodule_forward = forward_module(loads)
                sys.modules[".".join([full_name, submodule_name])] = submodule_forward
                if main_needs_multiload or is_forwarding:
                    for i in range(num_copies):
                        with multiload_context(i):
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
            for i in range(num_copies):
                if i == outer_context_id:
                    assert not hasattr(desired_module, "_parla_load_id")
                    desired_module._parla_load_id = i
                    multiloads.append(desired_module)
                    continue
                with multiload_context(i):
                    new_load = builtin_import(name, glob, loc, fromlist, level)
                    found_parent = None
                    new_desired_module = new_load
                    for submodule_name in parts[1:]:
                        found_parent = new_desired_module
                        new_desired_module = getattr(new_desired_module, submodule_name)
                    assert found_parent is parent_module
                    assert not hasattr(new_desired_module, "_parla_load_id")
                    new_desired_module._parla_load_id = i
                    multiloads.append(new_desired_module)
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
    wrap_imports.set(True)
    yield
    wrap_imports.set(False)

