//
// Created by amp on 3/6/20.
//

#define _GNU_SOURCE
#include <stdlib.h>
#include <dlfcn.h>
#include <errno.h>

#include "virt_dlopen.h"

#include "preload_shim.h"

static __thread virt_dlopen_state current_state = VIRT_DLOPEN_STATE_INITIALIZER;

virt_dlopen_state virt_dlopen_get_state() {
    return current_state;
}

virt_dlopen_state virt_dlopen_swap_state(int enabled, Lmid_t lm) {
    virt_dlopen_state old = current_state;
    current_state.enabled = (char)enabled;
    current_state.lm = lm;
    return old;
}

PRELOAD_SHIM(void*, dlopen, (const char *filename, int flags)) {
    if (current_state.enabled) {
        // Clear the global flag since it is not supported by dlmopen.
        flags &= ~RTLD_GLOBAL;
        void* lib = dlmopen(current_state.lm, filename, flags);
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
