#!/bin/bash

#SBATCH --job-name=sar_test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal    # partition(large/normal/cpu) where you submit
#SBATCH --account=peilab       # only require for multiple projects

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required

LOG_DIR="/home/hchener/logs/work_dirs/"
mkdir -p -- "$LOG_DIR"

nohup /home/hchener/.conda/envs/chenhao_env/bin/python train.py > ${LOG_DIR}train.log 2>&1 &