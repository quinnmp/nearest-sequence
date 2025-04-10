#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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
export CUDA_HOME=/mmfs1/sw/cuda/12.3.2

# Set explicit paths for compilation
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=:/gscratch/weirdlab/quinn/nearest-sequence/video_action_learning/
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so
#export LD_PRELOAD=$CONDA_PREFIX/lib/libEGL.so:$LD_PRELOAD
#export LD_PRELOAD=$CONDA_PREFIX/lib/libGL.so:$LD_PRELOAD
# For Mujoco
unset LD_PRELOAD
export LD_PRELOAD=$CONDA_PREFIX/lib/libGL.so
export CUDA_LAUNCH_BLOCKING=1
rm data/*lookup*

#conda install conda=23.11.0
#conda install -n base -c conda-forge conda=25.3.0
#conda --version
#conda deactivate
#conda env remove -n nearest-sequence
#conda env create -f environment.yml

#python -u hdf5_to_rgb.py data/stack_task_D0/demo.hdf5 data/stack)
#python -u robosuite_test_crop.py config/env/stack_task_D0.yml
#python -u nn_sweep_epochs.py config/env/stack_task_D0_rgb.yml config/policy/bc_img.yml
#python -u nn_sweep_epochs.py config/env/stack_task_D0_rgb.yml config/policy/bc_img.yml
#python -u nn_eval.py config/env/stack_task_D0_rgb.yml config/policy/bc.yml
#python -u eval_model.py config/env/stack_task_D0_rgb.yml config/policy/bc_img.yml
#python -u nn_eval.py config/env/stack_task_D0_r3m.yml config/policy/bc_img.yml
#python -u nn_eval.py config/env/stack_task_D0_rgb.yml config/policy/bc_img.yml

task="square_task_D0"

python -u robosuite_state_to_rgb.py config/env/${task}.yml
python -u eval_model.py config/env/${task}_rgb.yml config/policy/bc_img_resnet.yml
python -u rgb_to_features.py data/${task}/100_rgb.pkl cond_models/resnet.pth
python -u split_data.py data/${task}/100_features.pkl
python -u nn_param_optimizer.py config/env/${task}_features.yml config/policy/ns_dan_bc.yml

#python -u nn_model_optimizer.py config/env/hopper_rgb.yml config/policy/bc_img.yml
#python -u state_to_rgb_array.py config/env/hopper.yml
#python -u state_to_rgb_array.py config/env/hopper_new.yml
#python -u nn_sweep_epochs.py config/env/stack_task_D0_dino.yml config/policy/bc_img.yml
#python -u nn_eval.py config/env/hopper.yml config/policy/bc.yml
#python -u nn_sweep_epochs.py config/env/stack_task_D0_dino.yml config/policy/bc_img.yml
#python -u opengl_test.py
#python -u state_to_tapir.py config/env/hopper.yml
#python -u nn_param_optimizer.py config/env/stack_task_D0_r3m.yml config/policy/ns_dan_bc.yml
#python -u robosuite_state_to_kpt.py config/env/coffee_task_D0.yml data/coffee_task_D0/100_proprio.pkl
#python -u nn_param_optimizer.py config/env/coffee_task_D0_kpt.yml config/policy/ns_dan_bc.yml
#python -u nn_sweep_epochs.py config/env/stack_task_D0.yml config/policy/bc.yml
#python -u nn_model_optimizer.py config/env/maze.yml config/policy/bc.yml
#python -u nn_eval.py config/env/coffee_task_D0_kpt.yml config/policy/bc.yml
#python -u nn_eval.py config/env/square_task_D0_kpt.yml config/policy/bc.yml
#python -u nn_model_optimizer.py config/env/pick_and_place_task_D0.yml config/policy/bc.yml
#python -u nn_eval.py config/env/coffee_task_D0_img.yml config/policy/dan_img.yml
#python -u state_to_tapir.py config/env/hopper.yml; python -u rgb_arrays_to_mp4.py data/hopper_tapir_test.pkl data/hopper_tapir_test.mp4
#conda install -c pytorch cudatoolkit
#conda install -c conda-forge make cmake
#kernprof -l -v nn_eval.py config/env/square_task_D0_kpt.yml config/policy/ns_dan_bc.yml
#python -u pkl_numpy_to_torch.py
#python -u tsne_distances.py
#python -u nn_eval.py config/env/hopper_mix.yml config/policy/bc_img.yml
#python -u rgb_arrays_to_mp4.py data/hopper_tapir_test.pkl data/hopper_tapir_test.mp4
#python -u minimal_render.py
