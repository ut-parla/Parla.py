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
#include "stub_library.h"

// Declare the init function for calling from the stubs.
static void __stub$init_all_stubs();

EOF

echo "// Declare the ifunc stubs along with their resolvers" >> "$STUBC"
generate _\$_STUB

echo "// Create the init function" >> "$STUBC"
echo "_\$_START_INIT(NULL)" >> "$STUBC"
generate _\$_LOAD_STUB
echo "_\$_END_INIT()" >> "$STUBC"
