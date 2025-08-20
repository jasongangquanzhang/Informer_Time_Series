#!/bin/bash

#SBATCH --account=dachxiu-external
#SBATCH --partition=standard_h100
#SBATCH --cpus-per-task=1         
#SBATCH --mem=10G       
#SBATCH --time=0-04:00:00   
#SBATCH --gres=gpu:1
#SBATCH --job-name=timesfm
#SBATCH --array=1-100
#SBATCH --mail-type=END
#SBATCH --mail-user=gangquan.zhang@mail.utoronto.ca

module unload cuda
module load cuda/12.8
echo "Array ID: $SLURM_ARRAY_TASK_ID"

srun /home/zshen10/miniconda3/envs/olympus/bin/python3 /home/zshen10/gangquan_zhang/Informer_Time_Series/TimesFM/TimesFM_Pythia/finetune_timesfm.py $SLURM_ARRAY_TASK_ID