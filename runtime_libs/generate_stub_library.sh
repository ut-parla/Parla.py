#! /bin/bash

SO="$1"
SONAME="\"$(basename "$SO")\""
if [ -n "$2" ]; then
  SONAME="$2"
fi
STUBSO="libstub_$(basename "$SO"| sed "s/^lib//")"
STUBC="$STUBSO.c"
STUBVERSIONSCRIPTTMP="$STUBSO.versionscript.tmp"
STUBVERSIONSCRIPT="$STUBSO.versionscript"

function extract_symbols() {
  readelf --wide --dyn-syms "$1" | \
    sed -n 's/[[:space:]]\+/\t/g;/^\(\t[^\t]\+\)\{6\}\t[[:digit:]]\+\t[^[:space:]]\+/p'
}

function get_symbol_name() {
    line="$1"
    echo "$line" | cut -f 8
}

function escape_symbol_name() {
    name="$1"
    echo "$name" | sed 's/[^[:alnum:]$]/_/g'
}

function split_symbol_name() {
#    global SYMBOL_NAME SYMBOL_VERSION
    name="$1"
    read -r SYMBOL_NAME SYMBOL_VERSION < <(echo "$name" | sed 's/@@/ /')
}

# Generate a macro call for each exported symbol
function generate() {
  i=0
  extract_symbols $SO | (while read line; do
      name="$(get_symbol_name "$line")"
      if [ "$name" = "_init" ] || [ "$name" = "_fini" ]; then
          continue
      fi
      escaped_name="$(escape_symbol_name "$name")"
      split_symbol_name "$name"
      echo "/* $(echo "$line" | tr '\t' ' ') */" >> "$STUBC"
      if [ -n "$SYMBOL_VERSION" ]; then
        macro='_$_STUB_VERSION'
        echo "$macro($SYMBOL_NAME, \"$SYMBOL_VERSION\", $escaped_name, \"$name\");" >> "$STUBC"
        echo "$SYMBOL_VERSION {};" >> "$STUBVERSIONSCRIPTTMP"
      else
        macro='_$_STUB'
        echo "$macro($SYMBOL_NAME);" >> "$STUBC"
      fi
      printf "\rGenerating $macro... %d $SYMBOL_NAME, $SYMBOL_VERSION        " $i
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

EOF
# // Build: gcc -shared $STUBC -o $STUBSO -ldl

cat>> "$STUBC" <<"EOF"
#include "stub_library.h"

EOF

echo "// Create the init function" >> "$STUBC"
echo '_$_INIT('"$SONAME"')' >> "$STUBC"
echo >> "$STUBC"

echo "// Declare the ifunc stubs along with their resolvers" >> "$STUBC"
generate

if [ -f "$STUBVERSIONSCRIPTTMP" ]; then
  sort -u "$STUBVERSIONSCRIPTTMP" > "$STUBVERSIONSCRIPT"
fi
