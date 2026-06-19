#!/bin/bash
#SBATCH --job-name=ner_finetuning        # name of job
#SBATCH -C h100                          # ask for h100 node
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks-per-node=1              # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --gres=gpu:1                     # number of GPUs per node
#SBATCH --cpus-per-task=24               # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)
#SBATCH --hint=nomultithread             # hyperthreading is deactivated
#SBATCH --time=20:00:00                  # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logs/log_out_ner_%j.out # name of output file
#SBATCH --error=logs/log_err_ner_%j.out  # name of error file

set -e

# Change to the script's directory so relative paths work
cd "$(dirname "$0")"

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


DATASET="${1:-medmentions}"
CONFIG="train_ner_${DATASET}.cfg"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config file not found: ${CONFIG}"
  exit 1
fi

echo "============================================================"
echo "Training dataset=${DATASET}
echo "============================================================"

srun python train_ner_edsnlp.py train "${CONFIG}"

echo "============================================================"
echo "Evaluating dataset=${DATASET}
echo "============================================================"

srun python train_ner_edsnlp.py evaluate "${CONFIG}"