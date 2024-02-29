#!/bin/sh
#BSUB -J torch_job
#BSUB -q hpc
#BSUB -R "rusage[mem=64MB]"
#BSUB -o submitfiles/logs/torch_job_out_%J.txt
#BSUB -e submitfiles/logs/torch_job_err_%J.txt
#BSUB -W 48:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate cgd
module load pandas
python3 experiments/run_SPCA.py 2 20 1
