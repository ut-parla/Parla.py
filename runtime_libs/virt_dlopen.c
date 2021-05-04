//
// Created by amp on 3/6/20.
//

#define _GNU_SOURCE
#include <stdlib.h>
#include <dlfcn.h>
#include <errno.h>

#include "virt_dlopen.h"
#include "log.h"

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "preload_shim.h"

static __thread virt_dlopen_state current_state = VIRT_DLOPEN_STATE_INITIALIZER;

virt_dlopen_state virt_dlopen_get_state() {
    return current_state;
}

virt_dlopen_state virt_dlopen_swap_state(char enabled, long int lm) {
    virt_dlopen_state old = current_state;
    current_state.enabled = (char)enabled;
    current_state.lm = lm;
    return old;
}

PRELOAD_SHIM(void*, dlopen, (const char *filename, int flags)) {
    if (current_state.enabled) {
        DEBUG("Loading %s (%x) into %ld", filename, flags, current_state.lm);
        void* lib = dlmopen(current_state.lm, filename, flags);
//        DEBUG("Loaded %p", lib);
        if (current_state.lm == LM_ID_NEWLM && lib != NULL) {
            int ret = dlinfo(lib, RTLD_DI_LMID, &current_state.lm);
            if (ret != 0) {
                int tmp = errno;
                dlclose(lib);
                errno = tmp;
                return NULL;
            }
        }
        return lib;
    } else {
        init_dlopen();
        return next_dlopen(filename, flags);
    }
}

/*
PRELOAD_SHIM(int, openat, (int dirfd, const char *pathname, int flags)) {
    printf("************* OPENAT %s\n", pathname);
    return next_openat(dirfd, pathname, flags);
}

PRELOAD_SHIM(int, openat64, (int dirfd, const char *pathname, int flags)) {
    printf("************* OPENAT64 %s\n", pathname);
    return next_openat64(dirfd, pathname, flags);
}


PRELOAD_SHIM(FILE*, fopen, (const char* filename, const char* mode)) {
    printf("************* fopen  %s\n", filename);
    return next_fopen(filename, mode);
}

PRELOAD_SHIM(int, open64, (const char *pathname, int flags)) {
    printf("************* OPEN64 %s\n", pathname);
    return next_open64(pathname, flags);
}
*/

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