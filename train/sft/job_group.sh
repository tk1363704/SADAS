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



