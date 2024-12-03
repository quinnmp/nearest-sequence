#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-task=8
#SBATCH --mem=1024G
#SBATCH --time=72:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence

conda activate nearest-sequence

# Force software rendering
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

stdbuf -o0 -e0 python -u nn_model_optimizer.py config/env/hopper.yml config/policy/hopper/ns_dan_bc.yml 

