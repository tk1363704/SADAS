#!/bin/bash

#SBATCH -t 7-00:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --partition=A100


# Memory usage (MB)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=300000

#SBATCH --mail-user=devin.hua@monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# IMPORTANT!!! check the job name!
#SBATCH -o %J-main_all.out
#SBATCH -e %J-main_all.err
#
#
#
#
#SBATCH -J llama2


#!/bin/bash
#SBATCH --job-name=atom-llama
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20000
#SBATCH --partition A100
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=devin.hua@monash.edu
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=gpua100

module load anaconda
export CONDA_ENVS=/nfsdata/data/devinh/envs
source activate $CONDA_ENVS/deepspeed
cd /nfsdata/data/devinh/GPT-Bargaining/train/sft
bash finetune_lora_bargaining.sh



