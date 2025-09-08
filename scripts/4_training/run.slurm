#!/bin/bash
#SBATCH --job-name=training       # name of job
#SBATCH -C h100                     # uncomment for gpu_p6 partition (80GB H100 GPU)
#SBATCH --ntasks=4                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=4          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:4                 # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=24          # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logs/log_out%j.out    # name of output file
#SBATCH --error=logs/log_err%j.out     # name of error file (here, in common with the output file)
 
# Cleans out the modules loaded in interactive and inherited by default 
module purge
 
# Loading of modules
module load arch/h100
module load pytorch-gpu/py3/2.3.1

# Train models
python scripts/4_training/train.py $SCRIPT_ARGS
echo SCRIPT FINISHED
