This directory contains a collection of example programs written in Parla.
They are designed to be simple and easy-to-understand rather than to maximize performance.
Each example has a description at the top of the file.

The `inner.py` example is a good place to start.


We provide the following:
source.sh - A source file to initialize enviornment variables to defaults
commands.md - A file listing single use instructions on how to run each
example.
launcher.py - A script to run the experiments used in the SC22 paper.


All scripts assume they are run from $PARLA_ROOT.

Example Usage:

source examples/source.sh
python examples/launcher.py --figures 10

