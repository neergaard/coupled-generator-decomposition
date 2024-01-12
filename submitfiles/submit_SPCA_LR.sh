#!/bin/sh
#BSUB -J real_fit_job
#BSUB -q hpc
#BSUB -R "rusage[mem=64MB]"
#BSUB -o submitfiles/LR_fit_job_out_%J.txt
#BSUB -e submitfiles/LR_fit_job_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 
source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate cgd
module load pandas
pip install .
python experiments/run_SPCA_LR.py
