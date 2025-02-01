#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40s

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com

echo "ID: ${SLURM_JOB_ID}"
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence

conda activate nearest-sequence

# Force software rendering
#export MUJOCO_GL=osmesa
#export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "SLURM GPUS: ${CUDA_VISIBLE_DEVICES}"
export GPUS=$CUDA_VISIBLE_DEVICES
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGL.so:$LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so:$CONDA_PREFIX/lib/libGL.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD
unset LD_PRELOAD

#python -u robosuite_state_to_img.py config/env/stack_task_D0.yml
#python -u state_to_img.py config/env/hopper.yml
#python -u opengl_test.py
python -u nn_model_optimizer.py config/env/hopper_img_keypoint.yml config/policy/bc_img.yml
#python -u nn_sweep_epochs.py config/env/hopper_img_keypoint.yml config/policy/bc_img.yml
#python -u nn_sweep_k.py config/env/stack_task_D0_img.yml config/policy/dan_img.yml
#python -u nn_eval.py config/env/stack_task_D0.yml config/policy/bc.yml
#python -u nn_eval.py config/env/hopper.yml config/policy/hopper/ns_dan_bc.yml
#python -u state_to_tapir.py config/env/hopper_new.yml
#python -u nn_eval.py config/env/hopper_img_keypoint.yml config/policy/bc_img.yml
#python -u hdf5_to_d4rl.py data/stack_task_D0/demo.hdf5 data/stack_task_D0/stack_task_D0_100.pkl
#python -u rgb_arrays_to_mp4.py data/hopper_tapir_test.pkl data/hopper_tapir_test.mp4
#python -u minimal_render.py
