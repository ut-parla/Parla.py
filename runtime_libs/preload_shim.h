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

#endif //NESTED_RUNTIMES_PRELOAD_SHIM_H
