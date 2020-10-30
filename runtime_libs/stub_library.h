#ifndef STUB_LIBRARY_H
#define STUB_LIBRARY_H

#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <assert.h>

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
    static void* _$_mangle_resolver(n)() { \
        void *lib = dlmopen(LM_ID_BASE, NULL, RTLD_LAZY); \
        if(lib == NULL) { perror("dlmopen failed"); abort(); } \
        _$_mangle_stub(n) = dlsym(lib, #n); assert(_$_mangle_stub(n)); \
        return _$_mangle_stub(n); } \
    int n() __attribute__((ifunc (_$_string(_$_mangle_resolver(n)))))
//        DEBUG("Binding ifunc symbol: %s = %p", #n, _$_mangle_stub(n)); \

/// Begin the init function for this stub library.
#define _$_START_INIT(so) // No longer needed
/// Load a stub inside the init function
#define _$_LOAD_STUB(n) // No longer needed
/// End the init function for this stub library.
#define _$_END_INIT() // No longer needed

#endif