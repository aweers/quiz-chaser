import cv2
import pytesseract
import numpy as np

img = cv2.imread('slices/1494.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blur = cv2.medianBlur(img_gray, 5)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
inv_simple = 255-cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 0, 200), (180, 10, 255))
cv2.imwrite("th_mask.png", mask)

nmask = (np.min(img, axis=-1) < 160) * 255
nmask2 = (np.max(img, axis=-1) > 60) * 255
cv2.imwrite("th_nmask.png", nmask)
cv2.imwrite("th_nmask2.png", nmask2)

mask_wrong = ((np.min(img, axis=-1) < 160) * 255).astype(np.uint8)
mask_correct = ((np.max(img, axis=-1) > 60) * 255).astype(np.uint8)

if mask_correct[156:217, 486:920].sum() < mask_wrong[156:217, 486:920].sum():
    cv2.imwrite("chosen.png", mask_correct[156:217, 486:920])
    print("correct")
else:
    cv2.imwrite("chosen.png", mask_wrong[156:217, 486:700])
    print("wrong")
    ans2 = pytesseract.image_to_string(mask_wrong[156:217, 486:920], lang='deu', config='--psm 7')
    print("Ans: " + ans2)
    src = mask_wrong[156:217, 486:920]
    # src = cv2.copyMakeBorder(mask_correct[156:217, 486:700], 0, 0, 40, 0, cv2.BORDER_CONSTANT, None, [255, 255, 255])
    cv2.imwrite("src.png", src)
    for i in range(14):
        if i in [0, 2]:
            continue
        print(f"{i}: " + pytesseract.image_to_string(src, lang='deu', config=f'--psm {i}'))



# cv2.imwrite("th.png", th)
# cv2.imwrite("th1.png", inv_simple)
# cv2.imwrite("th2.png", cv2.adaptiveThreshold(inv_simple, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
