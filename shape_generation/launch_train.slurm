#!/bin/bash
#SBATCH --job-name=dit-fold-l-1024-4      # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node (best to match number of gpus. e.g. 4 gpus on gpu80 nodes)
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --constraint=gpu80
#SBATCH --time=45:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

export MASTER_PORT=$(get_free_port)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load anaconda3/2024.10
conda activate pytorch3d

srun python train_and_eval.py --experiment_name moment_gpt \
    --category chair \
    --num_classes 1 \
    --niter 2000 \
    --n_c 1024 \
    --n_p 4 \
    --model_type 'GPT2' \
    --bs 128 \
    --use_tb \
    --use_ema