import cv2
import os
import numpy as np
import pytesseract
import ffmpeg
import time
from enum import Enum
import argparse
import glob
import subprocess

def sample_frames(url, output):
    # ffmpeg.input(url, r=1).filter('fps', fps='1/30').crop(1405, 240, 258, 798).output(os.path.join(output, "frame_%04d.png")).run(quiet=True)
    os.makedirs(output, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", url, "-vf", "fps=1,crop=1405:240:258:798", os.path.join(os.path.dirname(__file__), output, "frame_%04d.png")])


def bgr2hsl(bgr):
    bgr /= 255.0
    dominant_color = np.argmax(bgr)
    det = np.max(bgr) - np.min(bgr)
    if det == 0:
        h = np.float32(0)
    elif dominant_color == 0:
        h = 4 + (bgr[2] - bgr[1]) / det
    elif dominant_color == 1:
        h = 2 + (bgr[0] - bgr[2]) / det
    else:
        h = (bgr[1] - bgr[0]) / det
    h *= 60
    
    l = (np.min(bgr) + np.max(bgr)) / 2
    if det == 0:
        s = np.float32(0)
    elif l > 0.5:
        s = det / (np.max(bgr) + np.min(bgr))
    else:
        s = det / (2.0 - np.max(bgr) - np.min(bgr))

    return h.item(), s.item(), l.item()

def green_answer(image):
    correct_a = bgr2hsl(np.mean(image[210:215, 45:65], axis=(0,1)))
    correct_b = bgr2hsl(np.mean(image[210:215, 495:515], axis=(0,1)))
    correct_c = bgr2hsl(np.mean(image[210:215, 945:965], axis=(0,1)))

    def is_green(clr):
        return clr[0] > 128 and clr[0] < 138 and clr[1] > 0.2 and clr[1] < 0.32 and clr[2] > 0.56 and clr[2] < 0.69

    if is_green(correct_a):
        return 0
    if is_green(correct_b):
        return 1
    if is_green(correct_c):
        return 2
    return -1


def process_frames(location):
    c_files = len(glob.glob(os.path.join(location, "frame_*.png")))
    result = []
    for im_i in range(1, c_files + 1):
        temp_file = os.path.join(os.path.dirname(__file__), location, "frame_" + str(int(im_i )).zfill(4) + ".png")
        img = cv2.imread(temp_file)
        if img is None:
            print(f"Error reading {im_i}")
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0, sigmaY=0)
        sobel_h = cv2.Sobel(img_blur, dx=0, dy=1, ksize=5, ddepth=cv2.CV_64F)
        sobel_v = cv2.Sobel(img_blur, dx=1, dy=0, ksize=5, ddepth=cv2.CV_64F)

        horizontal_lines = np.mean(sobel_h[0:22, :]) + np.mean(sobel_h[144:160]) + np.mean(sobel_h[220:226])
        vertical_lines = np.mean(np.abs(sobel_v[:, :23])) + np.mean(np.abs(sobel_v[156:210, 468:480])) + np.mean(np.abs(sobel_v[156:210, 920:930])) + np.mean(np.abs(sobel_v[:, 1389:]))
        if horizontal_lines > 1000 and vertical_lines > 1000: # question with answers
            ans = green_answer(img)
            if ans > -1:
                question = pytesseract.image_to_string(img_gray[20:143, 25:1384], lang='deu')
                ans1 = pytesseract.image_to_string(img_gray[156:217, 18:471], lang='deu')
                ans2 = pytesseract.image_to_string(img_gray[156:217, 486:920], lang='deu')
                ans3 = pytesseract.image_to_string(img_gray[156:217, 937:1384], lang='deu')

                result.append((im_i, question, ans1, ans2, ans3, ans))
    return result

def to_file(results, file_name, src_name):
    with open(file_name, "a") as f:
        for r in results:
            f.write("=" * 20 + "\n")

            f.write(src_name + " at " + str(r[0]) + "\n")
            f.write("Question\n")
            f.write(r[1] + "\n")

            f.write("=" * 10 + "\n")
            f.write("Answer A\n")
            f.write(r[2] + "\n")

            f.write("=" * 10 + "\n")
            f.write("Answer B\n")
            f.write(r[3] + "\n")

            f.write("=" * 10 + "\n")
            f.write("Answer C\n")
            f.write(r[4] + "\n")

            f.write("=" * 10 + "\n")
            f.write(f"Correct answer: {r[5]}\n")

            f.write("=" * 20 + "\n")

def clean_up(location):
    files = glob.glob(os.path.join(location, "frame_*.png"))
    for f in files:
        os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()

    tmp_dir = "out"
    sample_frames(args.video, tmp_dir)
    res = process_frames(tmp_dir)
    to_file(res, "questions.txt", "vid1")
    clean_up(tmp_dir)
