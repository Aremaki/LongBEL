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
echo "Job ID: $SLURM_JOB_ID"


# Launch training
python scripts/4_training_decoder/build_oof_memory_dataset.py --model-name Llama-3.1-8B-Instruct --dataset-name EMEA --add-headers --generate-oof-predictions
echo "SCRIPT FINISHED"