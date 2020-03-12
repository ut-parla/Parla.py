//
// Created by amp on 3/8/20.
//

#ifndef NESTED_RUNTIMES_SUPERVISOR_WRAPPERS_H
#define NESTED_RUNTIMES_SUPERVISOR_WRAPPERS_H

#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>

/** Set an environment variable inside a context. */
int context_setenv (long int context, const char *name, const char *value, int overwrite);
/** Unset an environment variable inside a context. */
int context_unsetenv (long int context, const char *name);

/** Set the allowed affinity inside a context created with context_new. */
void context_affinity_override_set_allowed_cpus (long int context, size_t cpusetsize, const cpu_set_t *cpuset);
void context_affinity_override_set_allowed_cpus_py (long int context, size_t ncpus, const int *cpus);

/** Create a new context (dlmopen namespace) and initialize it for use in parla. */
long int context_new();
/** Load a new library into a namespace using the standard mode (lazy with symbols available to other
 *  loads in this namespace). */
void * context_dlopen(long int context, const char*file);

#endif //NESTED_RUNTIMES_SUPERVISOR_WRAPPERS_H
