#!/bin/bash

#SBATCH --account=pi-dachxiu
#SBATCH --time=1-12:00:00
#SBATCH --job-name=TimesFm
#SBATCH --partition=amd
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --array=1-100
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=END
#SBATCH --mail-user=gangquan.zhang@mail.utoronto.ca

module load python/anaconda-2023.09
source activate olympus
echo "Array ID: $SLURM_ARRAY_TASK_ID"

srun /home/gangquanz/.conda/envs/olympus/bin/python3.11 /home/gangquanz/Informer_Time_Series/TimesFM/run_TimesFM.py $SLURM_ARRAY_TASK_ID
