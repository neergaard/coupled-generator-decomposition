#!/bin/sh
#BSUB -J real_fit_job
#BSUB -q hpc
#BSUB -R "rusage[mem=1GB]"
#BSUB -o real_fit_job_out_%J.txt
#BSUB -e real_fit_job_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
module load python3/3.10.7
module load numpy/1.23.3-python-3.10.7-openblas-0.3.21
python3 -m pip install --user tqdm torch
cd ..
pip install .
python3 experiments/run_SPCA_noregu.py 2 20
