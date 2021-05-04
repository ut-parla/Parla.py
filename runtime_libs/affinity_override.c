//
// Created by amp on 3/8/20.
//

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "preload_shim.h"
#include "log.h"

static cpu_set_t allowed_cpus = {-1, };

static int pid_mutilation_number = 0;

void set_pid_mutilation_number(int i) {
    pid_mutilation_number = i;
}

void affinity_override_set_allowed_cpus(size_t cpusetsize, const cpu_set_t *cpuset) {
    assert(cpusetsize <= sizeof(cpu_set_t));
//    for (size_t i = 0; i != cpusetsize; i++) {
//        ((char*)&allowed_cpus)[i] = ((char*)cpuset)[i];
//    }
    memcpy(&allowed_cpus, cpuset, cpusetsize);
}

static inline cpu_set_t affinity_override_restrict_cpu_set(size_t cpusetsize, const cpu_set_t *cpuset) {
    assert(cpusetsize <= sizeof(cpu_set_t));
    cpu_set_t tmp;
    CPU_ZERO(&tmp);
    CPU_AND_S(cpusetsize, &tmp, cpuset, &allowed_cpus);
    return tmp;
}

PRELOAD_SHIM(int, pthread_create, (pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg)) {
    init_pthread_create();
    int ret = next_pthread_create(thread, attr, start_routine, arg);
    if (ret == 0) {
//        DEBUG("Setting CPU affinity for new thread: %p", (void*)*thread);
        pthread_setaffinity_np(*thread, sizeof(allowed_cpus), &allowed_cpus);
    }
    return ret;
}

PRELOAD_SHIM(int, pthread_setaffinity_np, (pthread_t thread, size_t cpusetsize, const cpu_set_t *cpuset)) {
    cpu_set_t tmp = affinity_override_restrict_cpu_set(cpusetsize, cpuset);
    init_pthread_setaffinity_np();
    return next_pthread_setaffinity_np(thread, cpusetsize, &tmp);
}

PRELOAD_SHIM(int, sched_setaffinity, (pid_t pid, size_t cpusetsize, const cpu_set_t *mask)) {
    cpu_set_t tmp = affinity_override_restrict_cpu_set(cpusetsize, mask);
    init_sched_setaffinity();
    return next_sched_setaffinity(pid, cpusetsize, &tmp);
}

PRELOAD_SHIM(long, sysconf, (int name)) {
    init_sysconf();
    long ret = next_sysconf(name);
    if (name == _SC_NPROCESSORS_CONF || name == _SC_NPROCESSORS_ONLN)
        return MIN(ret, CPU_COUNT(&allowed_cpus));
    else
        return ret;
}

PRELOAD_SHIM(int, get_nprocs, (void)) {
    init_get_nprocs();
    return MIN(next_get_nprocs(), CPU_COUNT(&allowed_cpus));
}

PRELOAD_SHIM(int, get_nprocs_conf, (void)) {
    init_get_nprocs_conf();
    return MIN(next_get_nprocs_conf(), CPU_COUNT(&allowed_cpus));
}

PRELOAD_SHIM(pid_t, getpid, (void)) {
    init_getpid();
    pid_t pid = next_getpid();
    pid_t p = pid - pid_mutilation_number * 1111;
    if (pid != p)
        DEBUG("Lying about PID: Real pid %d, returning %d", pid, p);
    return p;
}

/*
PRELOAD_SHIM(int, open, (const char *pathname, int flags, mode_t mode)) {
    init_open();
    const char* vecid = getenv("VECID");
    printf("open shim:: VEC (%s) opening file %s\n", 
        (vecid!=NULL)? vecid : "NULL",
        pathname);

    //1. vecid must be in env
    //2. file must exist
    if (!strcmp("/proc/cpuinfo", pathname) && vecid) {
        char fpath[100];
        sprintf(fpath, "%s/%s%s", "/tmp/parla/fakecpuinfos", "cpuinfo_", vecid);
        
        // can't use O_RDONLY because it's defined in fcntl.h and including that
        //messes up the shim since it apparently redefines open
        int fid = next_open(fpath, 0, 0);
        if (fid < 0)
            printf("open shim: tried opening fake cpuinfo %s but failed. perhaps set_allowed_cpus wasn't called\n", fpath);
        else {
            printf("open shim: fake cpuinfo for VEC #%s locked and loaded\n", vecid);
            return fid;
        }
    }
    return next_open(pathname, flags, mode);
}
*/
