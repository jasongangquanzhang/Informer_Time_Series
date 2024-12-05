#!/bin/bash

#SBATCH --account=pi-dachenxiu
#SBATCH --mem=10G
#SBATCH --time=0-12:00:00
#SBATCH --job-name=enverus
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --array=1-100

module load python/booth/3.8

echo "Array ID: $SLURM_ARRAY_TASK_ID"

srun python3 /home/gangquanz/Informer_Time_Series/run_mercury.py $SLURM_ARRAY_TASK_ID
