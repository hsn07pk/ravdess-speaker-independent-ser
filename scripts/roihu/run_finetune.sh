#!/bin/bash
#SBATCH --job-name=ser_ft
#SBATCH --account=project_2020210
#SBATCH --partition=gpumedium
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:1
#SBATCH --mem=217086
#SBATCH --array=1-24%4
#SBATCH -o ./slurm_log/out_%A_%a.txt
#SBATCH -e ./slurm_log/err_%A_%a.txt

module load python-pytorch/2.10
export HF_HOME=/scratch/project_2020210/msohail/hf
export HF_HUB_OFFLINE=1

srun python finetune.py --fold $SLURM_ARRAY_TASK_ID --data /scratch/project_2020210/msohail/ravdess --out ./results --epochs 10 --bs 8
