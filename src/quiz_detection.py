import cv2
import os
import numpy as np
import pytesseract

images = [
        171, 172, 197, 218, 219, 245, 263, 278, 291, 298, 304,
        56, 89, 192, 212, 235, 260, 283, 310
        ]

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


# for im_i in images:
for im_i in range(1, 1333):
    temp_file = os.path.join(os.path.dirname(__file__), "out", "frame_" + str(int(im_i )).zfill(4) + ".png")
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
            print(f"Q: {question} (Correct: {ans}) \nA1: {ans1}\nA2: {ans2}\nA3: {ans3}")
            # out_file = os.path.join(os.path.dirname(__file__), "out", "out" + str(int(im_i)).zfill(4) + ".png")
            # cv2.imwrite(out_file, sobel_v)

