#!/bin/bash -x
#SBATCH --exclusive
#SBATCH --job-name=SCORES
#SBATCH --output=/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/bias_rmse_%A_%a.out
#SBATCH --error=/mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/bias_rmse_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=PESQ2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

cd $SLURM_SUBMIT_DIR

source /home/carlos.bastarz/miniconda3/etc/profile.d/conda.sh
conda activate SMNA-Exps

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python /mnt/beegfs/carlos.bastarz/SMNA_v3.0.x_check/anls_compare/pos/convert_to_netcdf/output/bias_rmse.py
