#!/bin/bash
#SBATCH --job-name=modelnet40_cls   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

module purge
module load anaconda3/2024.10
conda activate pytorch3d

srun python main.py --model MomentNet --epoch=100 --loss cross_entropy --num_points 2048

# srun python main.py --epoch=100 --loss cross_entropy
# srun python main.py --epoch=100 --loss cross_entropy+orthogonality
