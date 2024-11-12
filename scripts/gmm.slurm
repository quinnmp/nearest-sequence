#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0
#SBATCH -o log/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=quinnpfeifer@icloud.com
source ~/.bashrc

conda activate base

# Force software rendering
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python nn_push_t_param_optimizer.py config/push_t.yml
# python nn_eval.py config/hopper.yml
