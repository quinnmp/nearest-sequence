import pickle
import cv2
import numpy as np

def create_4x4_video(videos, output_path='results/4x4_eval_video.mp4'):
    """
    Create a 4x4 grid video from individual video frame lists
    """
    # Determine the size of the grid and individual videos
    grid_rows, grid_cols = 4, 4
    max_frames = max(len(video) for video in videos)
    
    # Determine frame size (assume all frames are the same size)
    frame_height, frame_width = videos[0][0].shape[:2]
    grid_frame_width = frame_width // 2
    grid_frame_height = frame_height // 2

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                          (grid_frame_width * grid_cols, grid_frame_height * grid_rows))

    # Iterate through frames
    for frame_idx in range(max_frames):
        # Create a blank grid frame
        grid_frame = np.zeros((grid_frame_height * grid_rows, 
                               grid_frame_width * grid_cols, 3), dtype=np.uint8)

        # Fill the grid with frames from each evaluation
        for i in range(grid_rows):
            for j in range(grid_cols):
                idx = i * grid_cols + j
                video = videos[idx]
                
                # Get the frame (loop last frame if not enough frames)
                frame = video[min(frame_idx, len(video)-1)]
                
                # Resize frame
                resized_frame = cv2.resize(frame, (grid_frame_width, grid_frame_height))
                
                # Place in grid
                grid_frame[
                    i*grid_frame_height:(i+1)*grid_frame_height, 
                    j*grid_frame_width:(j+1)*grid_frame_width
                ] = resized_frame

        # Write the grid frame
        out.write(grid_frame)

    # Release the video writer
    out.release()
    print(f"4x4 video saved to {output_path}")

# Load videos
videos = []
for i in range(16):
    videos.append(pickle.load(open(f"data/trial_{i + 1}_video", 'rb')))

# Create 4x4 video
create_4x4_video(videos)
