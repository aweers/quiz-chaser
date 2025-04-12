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

SRC_WIDTH = 960
def c_w(w):
    return int(w * SRC_WIDTH / 1920)

SRC_HEIGHT = 540
def c_h(h):
    return int(h * SRC_HEIGHT / 1080)

def probe_stream(url):
    res = ffmpeg.probe(url)

    def extract(entry):
        return {
                'index': entry['index'],
                'height': entry['height'],
                'width': entry['width']
                }
    details = [extract(e) for e in res['streams'] if e['codec_type'] == 'video']
    return details

def select_stream(details, quality):
    target_height = {
        'low': [360, 540, 720, 1080],
        'high': [1080, 720, 540, 360]
    }
    for target in target_height[quality]:
        for detail in details:
            if detail['height'] == target:
                return detail
    raise Exception()

def process_stream(url, stream_index=0):
    proc = ffmpeg.input(url)[f"0:{stream_index}"].filter('fps', fps='2').output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(SRC_WIDTH, SRC_HEIGHT)).run_async(pipe_stdout=True)
    count =0
    skip = 0
    result = []
    while True:
        in_bytes = proc.stdout.read(SRC_HEIGHT * SRC_WIDTH * 3)
        if not in_bytes:
            break
        if skip > 0:
            skip -= 1
            continue
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([SRC_HEIGHT, SRC_WIDTH, 3])
        )
        # in_frame = in_frame[c_w(1405):c_h(240), c_w(258):c_h(798)]
        in_frame = in_frame[c_h(798):c_h(1038), c_w(258):c_w(1663)]
        img = in_frame[..., ::-1].copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0, sigmaY=0)
        sobel_h = cv2.Sobel(img_blur, dx=0, dy=1, ksize=5, ddepth=cv2.CV_64F)
        sobel_v = cv2.Sobel(img_blur, dx=1, dy=0, ksize=5, ddepth=cv2.CV_64F)

        horizontal_lines = np.mean(sobel_h[c_h(0):c_h(22), :]) + np.mean(sobel_h[c_h(144):c_h(160)]) + np.mean(sobel_h[c_h(220):c_h(226)])
        vertical_lines = np.mean(np.abs(sobel_v[:, :c_w(23)])) + np.mean(np.abs(sobel_v[c_h(156):c_h(210), c_w(468):c_w(480)])) + np.mean(np.abs(sobel_v[c_h(156):c_h(210), c_w(920):c_w(930)])) + np.mean(np.abs(sobel_v[:, c_w(1389):]))
        if horizontal_lines > 1000 and vertical_lines > 1000: # question with answers
            ans = green_answer(img)
            if ans > -1:
                question = pytesseract.image_to_string(img_gray[c_h(20):c_h(143), c_w(25):c_w(1384)], lang='deu')
                ans1 = pytesseract.image_to_string(img_gray[c_h(156):c_h(217), c_w(18):c_w(471)], lang='deu')
                ans2 = pytesseract.image_to_string(img_gray[c_h(156):c_h(217), c_w(486):c_w(920)], lang='deu')
                ans3 = pytesseract.image_to_string(img_gray[c_h(156):c_h(217), c_w(937):c_w(1384)], lang='deu')

                result.append((count, question, ans1, ans2, ans3, ans))
        else:
            skip = 1

        count +=1

    return result

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
    correct_a = bgr2hsl(np.mean(image[c_h(210):c_h(215), c_w(45):c_w(65)], axis=(0,1)))
    correct_b = bgr2hsl(np.mean(image[c_h(210):c_h(215), c_w(495):c_w(515)], axis=(0,1)))
    correct_c = bgr2hsl(np.mean(image[c_h(210):c_h(215), c_w(945):c_w(965)], axis=(0,1)))

    def is_green(clr):
        return clr[0] > 128 and clr[0] < 138 and clr[1] > 0.2 and clr[1] < 0.32 and clr[2] > 0.56 and clr[2] < 0.69

    if is_green(correct_a):
        return 0
    if is_green(correct_b):
        return 1
    if is_green(correct_c):
        return 2
    return -1


def to_file(results, file_name):
    with open(file_name, "a") as f:
        for r in results:
            f.write("$$$$" + "\n")

            f.write("Question\n")
            f.write(r[1] + "\n")

            f.write("$$$" + "\n")
            f.write("Answer A\n")
            f.write(r[2] + "\n")

            f.write("$$$" + "\n")
            f.write("Answer B\n")
            f.write(r[3] + "\n")

            f.write("$$$" + "\n")
            f.write("Answer C\n")
            f.write(r[4] + "\n")

            f.write("$$$" + "\n")
            f.write(f"Correct answer: {r[5]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('output')
    args = parser.parse_args()

    streams = probe_stream(args.video)
    high_q_stream = select_stream(streams, 'low')
    SRC_HEIGHT = high_q_stream['height']
    SRC_WIDTH = high_q_stream['width']
    res = process_stream(args.video, high_q_stream['index'])
    to_file(res, args.output)
