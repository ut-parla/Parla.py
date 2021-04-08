#!/bin/bash
#SBATCH -J qr_fac_parla     # job name
#SBATCH -o experiment.out   # log file
#SBATCH -N 1                # total number of nodes
#SBATCH -n 16               # total number of mpi tasks requested
#SBATCH -p rtx              # queue (partition)
#SBATCH -t 03:00:00         # run time (hh:mm:ss)

echo -e "\n========================= SLURM JOB START =========================\n"

module load cuda

# Load my conda environment
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

export OMP_NUM_THREADS=16

echo -n '"iteration",' >> experiment.csv 
echo -n '"t1 GPU tasks","t1 H2D sum","t1 CPU ker sum","t1 GPU ker sum","t1 D2H sum","t1 total",' >> experiment.csv 
echo -n '"t2 total",' >> experiment.csv 
echo -n '"t3 GPU tasks","t3 H2D sum","t3 CPU ker sum","t3 GPU ker sum","t3 D2H sum","t3 total",' >> experiment.csv 
echo    '"iteration total"' >> experiment.csv 

set -v

## CPU 400k runs
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p cpu     -r  400000 -b 100000      >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p cpu     -r  400000 -b  25000      >> experiment.csv
#
## CPU 1.6M runs
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p cpu     -r 1600000 -b 400000      >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p cpu     -r 1600000 -b 100000      >> experiment.csv
#
## GPU 400k runs
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r  400000 -b 100000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r  400000 -b 100000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r  400000 -b 100000 -g 4 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r  400000 -b  25000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r  400000 -b  25000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r  400000 -b  25000 -g 4 >> experiment.csv
#
## GPU 1.6M runs
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r 1600000 -b 400000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r 1600000 -b 400000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r 1600000 -b 400000 -g 4 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r 1600000 -b 100000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r 1600000 -b 100000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2       --csv -p gpu     -r 1600000 -b 100000 -g 4 >> experiment.csv
#
## CPU + GPU 400k runs
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r  400000 -b 100000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r  400000 -b 100000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r  400000 -b 100000 -g 4 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r  400000 -b  25000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r  400000 -b  25000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r  400000 -b  25000 -g 4 >> experiment.csv
#
## CPU + GPU 1.6M runs
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r 1600000 -b 400000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r 1600000 -b 400000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r 1600000 -b 400000 -g 4 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r 1600000 -b 100000 -g 1 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r 1600000 -b 100000 -g 2 >> experiment.csv
#python qr_parla.py -c 1000 -i 5 -w 2 -t 16 --csv -p both    -r 1600000 -b 100000 -g 4 >> experiment.csv

# "Pure" GPU 400k runs
python qr_parla.py -c 1000 -i 5 -w 2       --csv -p puregpu -r  400000           -g 1 >> experiment.csv
python qr_parla.py -c 1000 -i 5 -w 2       --csv -p puregpu -r  400000           -g 2 >> experiment.csv
python qr_parla.py -c 1000 -i 5 -w 2       --csv -p puregpu -r  400000           -g 4 >> experiment.csv

# "Pure" GPU 1.6M runs
python qr_parla.py -c 1000 -i 5 -w 2       --csv -p puregpu -r 1600000           -g 4 >> experiment.csv

set +v

echo -e "\n========================== SLURM JOB END ==========================\n"
