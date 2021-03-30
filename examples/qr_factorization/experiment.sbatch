#!/bin/bash
#SBATCH -J qr_fac_parla     # job name
#SBATCH -o experiment.out
#SBATCH -N 1                # total number of node
#SBATCH -n 16               # total number of mpi tasks requested
#SBATCH -p rtx              # queue (partition) — normal, development, etc.
#SBATCH -t 03:00:00         # run time (hh:mm:ss) - 1.5 hours

echo -e "\n========================= SLURM JOB START =========================\n"

module load cuda

__conda_setup="$('/home1/07999/stephens/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home1/07999/stephens/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home1/07999/stephens/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home1/07999/stephens/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate parla

set -v

# CPU runs
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p cpu  -r  500000 -b 125000      >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p cpu  -r  500000 -b  31250      >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p cpu  -r 2000000 -b 500000      >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p cpu  -r 2000000 -b 125000      >> experiment.csv

# GPU 500k runs
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r  500000 -b 125000 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r  500000 -b 125000 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r  500000 -b 125000 -g 4 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r  500000 -b  31250 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r  500000 -b  31250 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r  500000 -b  31250 -g 4 >> experiment.csv

# GPU 2M runs
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r 2000000 -b 500000 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r 2000000 -b 500000 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r 2000000 -b 500000 -g 4 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r 2000000 -b 125000 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r 2000000 -b 125000 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p gpu  -r 2000000 -b 125000 -g 4 >> experiment.csv

# CPU + GPU 500k runs
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r  500000 -b 125000 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r  500000 -b 125000 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r  500000 -b 125000 -g 4 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r  500000 -b  31250 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r  500000 -b  31250 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r  500000 -b  31250 -g 4 >> experiment.csv

# CPU + GPU 2M runs
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r 2000000 -b 500000 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r 2000000 -b 500000 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r 2000000 -b 500000 -g 4 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r 2000000 -b 125000 -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r 2000000 -b 125000 -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 6 -t 16 --csv -p both -r 2000000 -b 125000 -g 4 >> experiment.csv

set +v

echo -e "\n========================== SLURM JOB END ==========================\n"
