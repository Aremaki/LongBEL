#!/bin/bash
#SBATCH --job-name=training       # name of job
#SBATCH -C h100                     # uncomment for gpu_p6 partition (80GB H100 GPU)
#SBATCH --nodes=1                    # nombre de noeud
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=24          # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logs/log_out%j.out    # name of output file
#SBATCH --error=logs/log_err%j.out     # name of error file (here, in common with the output file)
 
set -e
mkdir -p logs

# Clean modules and load environment
module purge
module load arch/h100
module load pytorch-gpu/py3/2.3.1

export OMP_NUM_THREADS=1

# Diagnostics
echo "Running on host: $(hostname)"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Job ID: $SLURM_JOB_ID"
echo "Starting training..."

# Launch training
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
torchrun --nproc_per_node=$GPUS_PER_NODE \
         --master-addr=$(hostname) \
         --master-port=$((10000 + $RANDOM % 50000)) \
         scripts/4_training_decoder/train.py $SCRIPT_ARGS

echo "SCRIPT FINISHED"