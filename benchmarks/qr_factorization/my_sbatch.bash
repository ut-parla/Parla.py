#!/bin/bash

rm -f experiment.csv experiment.out
touch experiment.csv
touch experiment.out
sbatch experiment.sbatch
tail -f experiment.out
