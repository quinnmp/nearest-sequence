#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=weirdlab
#SBATCH --partition=gpu-l40

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=100
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

echo "Starting optimization for hopper_5.yml..."
python nn_param_optimizer.py config/env/hopper_5.yml config/policy/hopper_5/ns_dan_bc.yml
echo "Finished optimization for hopper_5.yml."

echo "Starting optimization for hopper_10.yml..."
python nn_param_optimizer.py config/env/hopper_10.yml config/policy/hopper_10/ns_dan_bc.yml
echo "Finished optimization for hopper_10.yml."

echo "Starting optimization for hopper_15.yml..."
python nn_param_optimizer.py config/env/hopper_15.yml config/policy/hopper_15/ns_dan_bc.yml
echo "Finished optimization for hopper_15.yml."

echo "Starting optimization for hopper_20.yml..."
python nn_param_optimizer.py config/env/hopper_20.yml config/policy/hopper_20/ns_dan_bc.yml
echo "Finished optimization for hopper_20.yml."

echo "Starting optimization for hopper_25.yml..."
python nn_param_optimizer.py config/env/hopper_25.yml config/policy/hopper_25/ns_dan_bc.yml
echo "Finished optimization for hopper_25.yml."

