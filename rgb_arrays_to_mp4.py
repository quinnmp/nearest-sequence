import pickle
import cv2
import numpy as np
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description="Convert pickle frames to an MP4 video.")
parser.add_argument("pkl_path", type=str, help="Path to the pickle file containing video frames")
parser.add_argument("output", type=str, help="Output MP4 file name (e.g., output_video.mp4)")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video (default: 30)")
args = parser.parse_args()

# Load the pickle file
with open(args.pkl_path, "rb") as f:
    data = pickle.load(f)

# Video properties
frame_height, frame_width, channels = data[0].shape
output_filename = args.output

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' codec for MP4 videos
out = cv2.VideoWriter(output_filename, fourcc, args.fps, (frame_width, frame_height))

# Write frames to the video file
for frame in data:
    # Ensure frame is in the correct format (uint8)
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 255)).astype(np.uint8)
    out.write(frame)

# Release the VideoWriter
out.release()
print(f"Video saved as {output_filename}")

