import argparse

import cv2
import numpy as np
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True,
                    help="Path to the image to be scanned, language (ara or eng)")
parser.add_argument("--lang", required=True,
                    help="the desired language (ara or eng)")

args = parser.parse_args()

img = cv2.imread(args.image)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

final_img = None
screenCnt = None
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

# if no contour was found it will try to detect blacks
# as text ( if only a part of the document was pictured)
if screenCnt is None:
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
else:
    # show the contour (outline) of the piece of paper
    angle = cv2.minAreaRect(screenCnt)[-1]


    rottt = imutils.rotate_bound(img, angle=-angle)

    gray = cv2.cvtColor(rottt, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    _, thresh = cv2.threshold(edged, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, 1, 2)
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[:1][0]

    x, y, w, h = cv2.boundingRect(cnt)
    crop_img = rottt[y:y + h, x:x + w]
    final_img = crop_img

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle

# otherwise, just take the inverse of the angle to make
# it positive
if angle < -45:
    angle = -(90 + angle)

else:
    angle = -angle

# rotate the image to deskew it
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# show the output image
cv2.imshow("Input", img)
if final_img is None:
    final_img = rotated

cv2.imshow("final", final_img)
key = cv2.waitKey(0)
while key != ord(" "):
    if key == ord("d"):
        final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    elif key == ord("a"):
        final_img = cv2.rotate(final_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # print(key)
    cv2.imshow('rotate', final_img)
    key = cv2.waitKey(0)

gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 60, 255, 1)
blur = cv2.medianBlur(thresh, 1)

with open("output.txt", "w", encoding='utf-8') as out:
    out.write(pytesseract.image_to_string(gray, lang=args.lang))

with open("noisyImages_output.txt", "w", encoding='utf-8') as out:
    out.write(pytesseract.image_to_string(blur, lang=args.lang))

