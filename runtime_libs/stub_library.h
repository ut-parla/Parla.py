#ifndef STUB_LIBRARY_H
#define STUB_LIBRARY_H

#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "log.h"

// Some magic preprocessor tricks
#define _$__string(n) #n
#define _$_string(n) _$__string(n)
#define _$__concat(a, b) a##b
#define _$_concat(a,b) _$__concat(a,b)

// Macros to generate names that are highly unlikely to conflict.
#define _$_mangle_stub(n) _$_concat(__stub$$,n)
#define _$_mangle_resolver(n) _$_concat(__resolver$$,n)

/// Declare a stub for a symbol, n.
#define _$_STUB(n) static void* _$_mangle_stub(n) = NULL; \
    static void* _$_mangle_resolver(n)() { __stub$init_all_stubs(); \
        DEBUG("Loading symbol: %s", #n); return _$_mangle_stub(n); } \
    int n() __attribute__((ifunc (_$_string(_$_mangle_resolver(n)))))

/// Begin the init function for this stub library.
#define _$_START_INIT(so) static void __stub$init_all_stubs() { \
    DEBUG("Opening library to stub. "); \
    static volatile int __stub$inited = 0; \
    if (__glibc_likely(__stub$inited)) return; \
    DEBUG("Opening library to stub."); \
    void *lib = dlmopen(LM_ID_BASE, so, RTLD_NOW); \
    if(lib == NULL) { perror("dlmopen failed"); abort(); }
/// Load a stub inside the init function
#define _$_LOAD_STUB(n) _$_mangle_stub(n) = dlsym(lib, #n)
/// End the init function for this stub library.
#define _$_END_INIT() __stub$inited = 1; }

#endif