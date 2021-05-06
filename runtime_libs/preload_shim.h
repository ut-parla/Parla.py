//
// Created by amp on 3/8/20.
//

#ifndef NESTED_RUNTIMES_PRELOAD_SHIM_H
#define NESTED_RUNTIMES_PRELOAD_SHIM_H

// Some magic preprocessor tricks
#define __STRING__(n) #n
#define _STRING(n) __STRING__(n)
#define __CONCAT__(a, b) a##b
#define _CONCAT(a,b) __CONCAT__(a,b)

#include <dlfcn.h>

/// Declare a wrapper/shim for function name with return type rt and arguments args (args should be in parens)
#define PRELOAD_SHIM(rt, name, args) \
    rt(* volatile _CONCAT(next_, name))args = NULL; \
    static inline void _CONCAT(init_, name)() { \
        if (_CONCAT(next_, name) == NULL) \
            _CONCAT(next_, name) = dlsym(RTLD_NEXT, _STRING(name)); } \
    rt name args

/*
// Declare a wrapper for function "name" with return type "rt" and arguments "args".
// version suffix is a unique suffix to append to name for the name of the
// new implementation. version is the symbol version of the symbol you're wrapping.
#define PRELOAD_SHIM_VERSIONED(rt, name, version_suffix, version, isdefault, args) \
     __asm__(".symver "_STRING(name) "_" _STRING(version_suffix) "_new," _STRING(name) DEFAULT_SPECIFIER(isdefault) "VER_2"); \
     rt(* volatile _CONCAT(next_, name))args = NULL; \
     static inline void _CONCAT(init_, name)() { \
         if (_CONCAT(next_, name) == NULL) \
             _CONCAT(next, name) = dlsym(RTLD_NEXT, _STRING(name)); } \
     rt name args
*/

#endif //NESTED_RUNTIMES_PRELOAD_SHIM_H
