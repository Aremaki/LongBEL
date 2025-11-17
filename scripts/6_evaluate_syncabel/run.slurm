#!/bin/bash
#SBATCH --job-name=evaluation_cpu
#SBATCH --partition=cpu_p1              # CPU partition on Jean Zay
#SBATCH --cpus-per-task=24              # number of CPU cores
#SBATCH --hint=nomultithread            # disable hyperthreading
#SBATCH --time=20:00:00                 # max execution time
#SBATCH --output=logs/log_out_%j.out
#SBATCH --error=logs/log_err_%j.err

# Clean environment
module purge

# Load CPU Python stack (standard)
module load cpuarch/amd
module load python/3.11.5

# (Optional) if you installed packages with conda, activate your env
# source ~/miniconda3/bin/activate your_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $SLURM_CPUS_PER_TASK CPU cores"
python -u scripts/6_evaluate_syncabel/evaluate.py $SCRIPT_ARGS

echo "SCRIPT FINISHED"
