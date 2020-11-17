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

/// Begin the init function for this stub library.
#define _$_INIT(so) static void* __stub$get_lib() { \
        static volatile void* __stub$lib = NULL; \
        if (__glibc_likely(__stub$lib != NULL)) return (void*)__stub$lib; \
        DEBUG("Opening library to stub: " #so); \
        __stub$lib = dlmopen(LM_ID_BASE, so, RTLD_LAZY); \
        if(__stub$lib == NULL) { perror("dlmopen failed"); abort(); } \
        return (void*)__stub$lib;\
    }

// TODO: Dedup the two STUB macros.

/// Declare a stub for a symbol, name, with symbol version, version.
#define _$_STUB_VERSION(name, version, escaped_name, name_with_version) \
    static void* _$_mangle_resolver(escaped_name)() { \
        static volatile void* _$_mangle_stub(escaped_name) = NULL; \
        if (_$_mangle_stub(escaped_name) == NULL) { \
            void *lib = __stub$get_lib(); \
            _$_mangle_stub(escaped_name) = dlvsym(lib, #name, version); \
            assert(_$_mangle_stub(escaped_name)); \
        } \
        return (void*)_$_mangle_stub(escaped_name); } \
    int _$_mangle_stub(escaped_name)() __attribute__((ifunc (_$_string(_$_mangle_resolver(escaped_name))))); \
    __asm__(".symver " _$_string(_$_mangle_stub(escaped_name)) ", " name_with_version)
// TODO: Once we only need to support new compilers use the following instead of inline ASM:
//   __attribute__ ((symver (name_with_version)))

/// Declare a stub for a symbol, escaped_name.
#define _$_STUB(escaped_name) \
    static void* _$_mangle_resolver(escaped_name)() { \
        static void* _$_mangle_stub(escaped_name) = NULL; \
        if (_$_mangle_stub(escaped_name) == NULL) { \
            void *lib = __stub$get_lib(); \
            _$_mangle_stub(escaped_name) = dlsym(lib, #escaped_name); \
            assert(_$_mangle_stub(escaped_name)); \
        } \
        return _$_mangle_stub(escaped_name); } \
    int escaped_name() __attribute__((ifunc (_$_string(_$_mangle_resolver(escaped_name)))));

#endif