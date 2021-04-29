//
// Created by amp on 1/22/20.
//

#ifndef NESTED_RUNTIMES_LOG_H
#define NESTED_RUNTIMES_LOG_H

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

#ifndef NDEBUG
#define DEBUG(str, ...) fprintf(stderr, "LOG: %s %s:%d " str "\n", __PRETTY_FUNCTION__, __FILE__, __LINE__, ## __VA_ARGS__)
#else
#define DEBUG(str, ...)
#endif

#define CHECK(p, s) if(!(p)) perror(s)

void *dlmopen_debuggable(long nsid, const char *file, int mode);

#endif //NESTED_RUNTIMES_LOG_H
