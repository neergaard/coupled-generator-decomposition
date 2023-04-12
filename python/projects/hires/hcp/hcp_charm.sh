#!/bin/bash

#SBATCH --job-name=hcp_charm    # Job name
#SBATCH --output=/mnt/newprojects/INN/jesper/nobackup/HiRes/slurm_logs/%A_%a.log          # A = master job id, a = task job id
#SBATCH --nodes=1                   # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --ntasks=1                  # Relevant when program implements MPI (multi system/distributed parallelism)
#SBATCH --cpus-per-task=2           # Relevant when program implements MP (single system parallelism, e.g., OpenMP, TBB)
#SBATCH --mem=10G                    # Job memory request
#SBATCH --array=1-1                # or 1,2,4,5,9 ; access as $SLURM_ARRAY_TASK_ID

echo "Job Information"
echo
echo "Job name     :  $SLURM_JOB_NAME"
echo "Job ID       :  $SLURM_ARRAY_JOB_ID"
echo "Task ID      :  $SLURM_ARRAY_TASK_ID"
echo "Cluster name :  $SLURM_CLUSTER_NAME"
echo "Node name    :  $SLURMD_NODENAME"
echo "Date         :  $(date)"
echo "Working dir  :  $SLURM_SUBMIT_DIR"
echo

source path.sh

subject=$(sed -n $SLURM_ARRAY_TASK_ID"p" $HCP_DATA/subjects.txt)
t1=$HCP_DATA/$subject/T1w/T1w_acpc_dc_restore.nii.gz
t2=$HCP_DATA/$subject/T1w/T2w_acpc_dc_restore.nii.gz

# By default, functions are not exported to be available in subshells so we
# need this before we can use 'conda activate'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simnibs

cd $HCP_CHARM
charm $subject $t1 $t2 --registerT2 --initatlas --segment --debug --forceqform --forcerun
