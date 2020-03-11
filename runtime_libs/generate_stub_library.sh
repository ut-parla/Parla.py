#! /bin/bash

SO="$1"
STUBSO="libstub_$(basename "$SO"| sed "s/^lib//")"
STUBC="$STUBSO.c"

function extract_symbols() {
  readelf --wide --dyn-syms "$1" | \
    sed -n 's/[[:space:]]\+/\t/g;/^\(\t[^\t]\+\)\{6\}\t[[:digit:]]\+/p'
}

function get_symbol_name() {
    line="$1"
    echo "$line" | cut -f 8
}

# Generate a macro call for each exported symbol
function generate() {
  macro="$1"
  i=0
  extract_symbols $SO | (while read line; do
      name="$(get_symbol_name "$line")"
      if [ "$name" = "_init" ] || [ "$name" = "_fini" ]; then
          continue
      fi
      echo "$macro($name); /* $(echo "$line" | tr '\t' ' ') */" >> "$STUBC"
      printf "\rGenerating $macro... %d " $i
      i=$((i+1))
    done)
  echo "Done."
}

truncate --size 0 "$STUBC"

cat >> "$STUBC" <<EOF
// AUTOMATICALLY GENERATED stub library for $SO
// DO NOT MODIFY.

// This stub library should be loaded inside a dlmopen namespace to forward calls
// to the original library in the base namespace.

// Build: gcc -shared $STUBC -o $STUBSO -ldl

EOF

cat>> "$STUBC" <<"EOF"
#define _GNU_SOURCE
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

// Some magic preprocessor tricks
#define _$__string(n) #n
#define _$_string(n) _$__string(n)
#define _$__concat(a, b) a##b
#define _$_concat(a,b) _$__concat(a,b)

// Macros to generate names that are highly unlikely to conflict.
#define _$_mangle_stub(n) _$_concat(__stub$$,n)
#define _$_mangle_resolver(n) _$_concat(__resolver$$,n)

/// Declare a stub for a symbol, n.
#define STUB(n) static void* _$_mangle_stub(n) = NULL; \
    static void* _$_mangle_resolver(n)() { __stub$init_all_stubs(); \
        return _$_mangle_stub(n); } \
    int n() __attribute__((ifunc (_$_string(_$_mangle_resolver(n)))))

/// Begin the init function for this stub library.
#define START_INIT(so) static void __stub$init_all_stubs() { \
    static volatile int __stub$inited = 0; \
    if (__glibc_likely(__stub$inited)) return; \
    void *lib = dlmopen(LM_ID_BASE, so, RTLD_LAZY|RTLD_GLOBAL); \
    if(lib == NULL) { perror("dlmopen failed"); abort(); }
/// Load a stub inside the init function
#define LOAD_STUB(n) _$_mangle_stub(n) = dlsym(lib, #n)
/// End the init function for this stub library.
#define END_INIT() __stub$inited = 1; }

// Declare the init function for calling from the stubs.
static void __stub$init_all_stubs();

EOF

echo "// Declare the ifunc stubs along with their resolvers" >> "$STUBC"
generate STUB

echo "// Create the init function" >> "$STUBC"
echo "START_INIT(NULL)" >> "$STUBC"
generate LOAD_STUB
echo "END_INIT()" >> "$STUBC"

echo "Build with:"
echo "gcc -shared $STUBC -o $STUBSO -ldl"
