#!/bin/bash

#SBATCH --account=pi-dachxiu
#SBATCH --time=1-12:00:00
#SBATCH --job-name=informer_TS
#SBATCH --partition=amd
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100
#SBATCH --mem-per-cpu=8000

module load python/anaconda-2023.09
source activate ganett

echo "Array ID: $SLURM_ARRAY_TASK_ID"

srun /home/gangquanz/.conda/envs/ganett/bin/python3.12 /home/gangquanz/Informer_Time_Series/run.py $SLURM_ARRAY_TASK_ID
