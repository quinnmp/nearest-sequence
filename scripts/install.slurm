#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --account=cse
#SBATCH --partition=gpu-rtx6k

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH -o log/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=quinnpfeifer@icloud.com

echo "ID: ${SLURM_JOB_ID}"
source ~/.bashrc
cd /gscratch/weirdlab/quinn/nearest-sequence

#conda deactivate
#conda deactivate
#conda env remove -n nearest-sequence
#conda env create -f environment.yml
conda activate nearest-sequence

#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/robomimic*
#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/robosuite*
#rm -rf $CONDA_PREFIX/lib/python3.10/site-packages/mimicgen*
export CUDA_HOME=/mmfs1/sw/cuda/12.3.2/
pip install --force-reinstall --no-cache-dir -e GroundingDINO
pip install -e ../cloned/mimicgen
pip install pynput==1.6.0
pip install -e ../cloned/robosuite
pip install -e ../cloned/robomimic
pip install mujoco==2.3.2
pip install numba==0.58.1


#ln -s $CONDA_PREFIX/lib/libEGL.so.1 $CONDA_PREFIX/lib/libEGL.so
#mkdir -p $CONDA_PREFIX/include/X11
#cp $CONDA_PREFIX/lib/python3.10/site-packages/egl_probe/glad/X11/*.h $CONDA_PREFIX/include/X11/
