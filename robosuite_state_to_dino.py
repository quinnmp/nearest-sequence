import pickle
import os

from nn_util import construct_env, crop_and_resize, frame_to_dino, get_semantic_frame_and_box, frame_to_obj_centric_dino, reset_vision_ob, get_proprio
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
import numpy as np
import gym
gym.logger.set_level(40)
from argparse import ArgumentParser
import yaml
from rgb_arrays_to_mp4 import rgb_arrays_to_mp4


parser = ArgumentParser()
parser.add_argument("env_config_path", help="Path to environment config file")
args, _ = parser.parse_known_args()

with open(args.env_config_path, 'r') as f:
    env_cfg = yaml.load(f, Loader=yaml.FullLoader)

data = pickle.load(open(env_cfg['demo_pkl'], 'rb'))

env = construct_env(env_cfg)
env_name = env_cfg['name']

camera_names = [env.env.sim.model.camera_id2name(i) for i in range(env.env.sim.model.ncam)]
camera_names = env_cfg['cams']
print(camera_names)

crops = env_cfg.get('crops', {})
print(crops)

height, width = 224, 224

img_data = []

def main():
    for traj in range(len(data)):
        frames = []
        print(f"Processing traj {traj}...")
        initial_state = dict(states=data[traj]['states'][0])
        initial_state["model"] = data[traj]["model_file"]
        traj_obs = []

        env.reset()
        env.reset_to(initial_state)
        reset_vision_ob()
        for i in range(len(data[traj]['actions'])):
            full_frame = np.empty((height, 0, 3), dtype=np.uint8)
            for camera in camera_names:
                crop_corners = np.array(crops.get(camera, [[0, 0], [width, height]]))

                crop_width = crop_corners[1][0] - crop_corners[0][0]
                render_width = width / crop_width

                crop_height = crop_corners[1][1] - crop_corners[0][1]
                render_height = height / crop_height

                frame = env.render(mode='rgb_array', height=round(render_height), width=round(render_width), camera_name=camera)
                assert frame is not None

                crop_corners[:, 0] *= render_width
                crop_corners[:, 1] *= render_height
                crop_corners = np.round(crop_corners).astype(np.uint16)
                cropped_frame = frame[crop_corners[0][1]:crop_corners[1][1], crop_corners[0][0]:crop_corners[1][0], :]

                full_frame = np.hstack((full_frame, cropped_frame))
            frames.append(full_frame)

            proprio_state = get_proprio(env_cfg, env.get_observation())
            obs = frame_to_dino(full_frame, proprio_state=proprio_state, numpy_action=False)
            traj_obs.append(obs.cpu().detach().numpy())

            env.step(data[traj]['actions'][i])

        # Sanity check - must actually be a success
        if env.get_reward() == 1:
            img_data.append({'observations': np.array(traj_obs), 'actions': data[traj]['actions']})
            rgb_arrays_to_mp4(frames, f"data/{traj}.mp4")
        else:
            print("REJECTING TRAJECTORY")

    print(f"Success! Dumping data to {env_cfg['demo_pkl'][:-4] + '_dino.pkl'}")
    pickle.dump(img_data, open(env_cfg['demo_pkl'][:-4] + '_dino.pkl', 'wb'))

main()
