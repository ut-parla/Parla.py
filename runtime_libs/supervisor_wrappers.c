//
// Created by amp on 3/8/20.
//

#define _GNU_SOURCE
#include <dlfcn.h>
#include <gnu/lib-names.h>
#include <sched.h>
#include <assert.h>

#include "supervisor_wrappers.h"
#include "log.h"

long int context_new() {
    void* dlh_environ = dlmopen_debuggable(LM_ID_NEWLM, "libparla_context.so", RTLD_NOW|RTLD_GLOBAL);
    if(!dlh_environ) return -1;
    Lmid_t lmid;
    int r = dlinfo(dlh_environ, RTLD_DI_LMID, &lmid);
    DEBUG("Loaded libparla_context.so into %ld (r=%d)", lmid, r);
    if (r != 0) return -1;
    return lmid;
}


void * context_dlopen(long int context, const char *file) {
    int flags = RTLD_LAZY|RTLD_GLOBAL;
    DEBUG("Loading %s (%x) into %ld", file, flags, context);
    void* p = dlmopen_debuggable(context, file, flags);
//    DEBUG("Loaded %p", p);
    return p;
}

// Some magic preprocessor tricks
#define __STRING__(n) #n
#define _STRING(n) __STRING__(n)
#define __CONCAT__(a, b) a##b
#define _CONCAT(a,b) __CONCAT__(a,b)

// TODO: This will be insanely slow since it loads the symbol for each call.

#define CONTEXT_CALLER(__soname, __rt, __name, __args, ...) \
    __rt _CONCAT(context_, __name) (long int context, ## __VA_ARGS__) { \
        void *__lib = dlmopen(context, __soname, RTLD_LAZY); \
        if(!__lib) DEBUG("call into context failed: %s", dlerror()); \
        __rt(* _CONCAT(p_, __name))(__VA_ARGS__) = dlsym(__lib, _STRING(__name)); \
        __rt __ret = _CONCAT(p_, __name)__args; \
        return __ret; }
//        dlclose(lib);
#define CONTEXT_CALLER_VOID(__soname, __rt, __name, __args, ...) \
    __rt _CONCAT(context_, __name) (long int context, ## __VA_ARGS__) { \
        void *__lib = dlmopen(context, __soname, RTLD_LAZY); \
        if(!__lib) DEBUG("call into context failed: %s", dlerror()); \
        __rt(* _CONCAT(p_, __name))(__VA_ARGS__) = dlsym(__lib, _STRING(__name)); \
        _CONCAT(p_, __name)__args; }

CONTEXT_CALLER(LIBC_SO, int, setenv, (name, value, overwrite), const char *name, const char *value, int overwrite)

CONTEXT_CALLER(LIBC_SO, int, unsetenv, (name), const char *name)

CONTEXT_CALLER_VOID("libparla_context.so", void, affinity_override_set_allowed_cpus, (cpusetsize, cpuset), size_t cpusetsize, const cpu_set_t *cpuset)

void context_affinity_override_set_allowed_cpus_py (Lmid_t context, size_t ncpus, const int *cpus) {
    cpu_set_t tmp;
    CPU_ZERO(&tmp);
    for (int i=0; i < ncpus; i++) {
        CPU_SET(cpus[i], &tmp);
    }
    context_affinity_override_set_allowed_cpus(context, sizeof(tmp), &tmp);
}

