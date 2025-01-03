#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence

conda activate nearest-sequence

# Force software rendering
# export MUJOCO_GL=osmesa
# export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD

#python -u robosuite_state_to_img.py config/env/square_task_D1.yml
#python -u nn_model_optimizer.py config/env/square_task_D1_img.yml config/policy/bc_img.yml
#python -u nn_sweep_k.py config/env/hopper_img.yml config/policy/dan_img.yml
python -u nn_eval.py config/env/square_task_D1_img.yml config/policy/bc_img.yml
#python -u hdf5_to_d4rl.py data/square_task_D1/demo.hdf5 data/square_task_D1/square_task_D1_1000.pkl
#python -u rgb_arrays_to_mp4.py data/trial_1_video data/trial_1_video.mp4
