//
// Created by amp on 2/20/20.
//

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdlib.h>
#include "log.h"

__attribute__((weak))
void *dlmopen_debuggable(long nsid, const char *file, int mode) {
    if (getenv("DEBUG_DLMOPEN")) {
        DEBUG("Using dlopen in place of dlmopen; should have had nsid=%ld but this is being ignored. (file=%s, mode=%x)", nsid, file, mode);
        return dlopen(file, mode);
    } else {
        return dlmopen(nsid, file, mode);
    }
}