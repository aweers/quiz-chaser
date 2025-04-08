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
    os.makedirs(output_dir, exist_ok=True)
    probe = ffmpeg.probe(video_path)
    video_probe = next((stream for stream in probe['streams'] if 'duration' in stream), None)
    print(video_probe)
    duration = float(video_probe['duration'])
    print(f"Duration of video file is {duration:.1f}s")

    current_time = offset
    state = State.OOQ
    
    counts = 10

    while current_time < duration:
        sample_interval = dense_interval if state == State.IN_QUESTION else inital_interval
        # with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
        temp_file = os.path.join(os.path.dirname(__file__), output_dir, str(current_time) + ".jpg")
        ffmpeg.input(video_src, ss=current_time).filter('scale', 640, -1).output(temp_file, vframes=1).run()
        counts -= 1
        if counts < 0:
            break

        frame = cv2.imread(temp_file)
        if frame is None:
            print(f"Error processing frame at {current_time}s")
            current_time += sample_interval
            continue
        current_time += sample_interval
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
    video_src = args.video
    sec = 500
    extract_questions(video_src, "out")
