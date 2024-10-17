import os
import subprocess

# Set the LD_LIBRARY_PATH
mujoco_path = "/home/quinn/.mujoco/mujoco210/bin"
current_ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
new_ld_library_path = f"{mujoco_path}:{current_ld_library_path}"
os.environ['LD_LIBRARY_PATH'] = new_ld_library_path

# Run Scalene with the updated environment
subprocess.run(["scalene", "--html", "model_optimizer.py", "config/coffee_pull_nn_gm.yml"])
