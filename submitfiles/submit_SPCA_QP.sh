#!/bin/sh
#BSUB -J QP_job
#BSUB -q hpc
#BSUB -R "rusage[mem=128MB]"
#BSUB -o submitfiles/logs/QP_job_out_%J.txt
#BSUB -e submitfiles/logs/QP_job_err_%J.txt
#BSUB -W 72:00 
#BSUB -n 8
#BSUB -R "span[hosts=1]"

# -- commands you want to execute -- 

source /dtu-compute/macaroni/miniconda3/bin/activate
conda activate cgd
module load pandas
python3 experiments/run_SPCA_QP.py 2 20
