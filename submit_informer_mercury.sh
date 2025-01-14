#!/bin/bash
#SBATCH --account=pi-dachxiu
#SBATCH --mem=10G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=informer_TS
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --array=1-100

module load python/booth/3.12
source ~/venv/gangquanz/bin/activate
echo "Array ID: $SLURM_ARRAY_TASK_ID"

srun python3 /home/gangquanz/Informer_Time_Series/run.py $SLURM_ARRAY_TASK_ID
