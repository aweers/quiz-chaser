import ffmpeg
import os
import cv2
import numpy as np
import tempfile
import time
from enum import Enum
import argparse

class State(Enum):
    OOQ = 0
    IN_QUESTION = 1

def extract_questions(video_path: str, output_dir: str, inital_interval:float = 3.0, dense_interval: float = 1.0, offset: int = 60):
    RESOLUTION = (960, 540)
    CROP = (int(258 * RESOLUTION[0] / 1920), int(798 * RESOLUTION[1] / 1080), int(1663 * RESOLUTION[0] / 1920), int(1038 * RESOLUTION[1] / 1080)) # ltrb at 1920x1080
    COLORS_QUESTION = [
            (5, 49, 28, 42, 54),
            (5, 59, 27, 41, 53),
            (680, 5, 0, 11, 30),
            (653, 76, 41, 47, 55)
    ]
    TOLERANCE = 5

    os.makedirs(output_dir, exist_ok=True)
    probe = ffmpeg.probe(video_path)
    video_probe = next((stream for stream in probe['streams'] if 'duration' in stream), None)
    print(video_probe)
    duration = float(video_probe['duration'])
    print(f"Duration of video file is {duration:.1f}s")

    current_time = offset
    state = State.OOQ
    
    counts = 100

    while current_time < duration:
        sample_interval = dense_interval if state == State.IN_QUESTION else inital_interval
        # with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
        # temp_file = os.path.join(os.path.dirname(__file__), output_dir, str(current_time) + ".png")
        temp_file = os.path.join(os.path.dirname(__file__), output_dir, "frame_" + str(int(current_time // 2)).zfill(4) + ".png")
        # ffmpeg.input(video_src, ss=current_time).crop(CROP[0], CROP[1], CROP[2]-CROP[0], CROP[3]-CROP[1]).output(temp_file, vframes=1).run()
        counts -= 1
        if counts < 0:
            break

        frame = cv2.imread(temp_file)
        if frame is None:
            print(f"Error processing frame at {current_time}s")
            current_time += sample_interval
            continue

        color_matched = []
        for y, x, r, g, b in COLORS_QUESTION:
            target_color = np.array([b, g, r])
            print(f"Target: {b}, {g}, {r}\t Frame: {frame[x, y]}")
            color_diff = np.sum(np.abs(target_color - frame[x, y]))
            if color_diff < TOLERANCE:
                color_matched.append(1)
            else:
                color_matched.append(0)

        print(f"Color result for {current_time}: {color_matched}")


        current_time += sample_interval
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
    video_src = args.video
    sec = 450
    extract_questions(video_src, "out", offset=sec)
