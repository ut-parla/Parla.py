#!/bin/bash

LD_LIBRARY_PATH="$HOME/VECs/Parla.py/runtime_libs/build:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" LD_PRELOAD="$HOME/VECs/Parla.py/runtime_libs/build/libparla_supervisor.so:/home/will/hacked-libc/lib/libSegFault.so" "$HOME/VECs/Parla.py/runtime_libs/usingldso" "/home/will/hacked-libc" python3 "$@"