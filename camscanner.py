import cv2
import numpy as np
import pytesseract
import imutils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread('arabic.png')

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height=500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", img)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

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
    angle = cv2.minAreaRect(coords)[-1]

# else
else:
    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    angle = cv2.minAreaRect(screenCnt)[-1]


# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
    angle = -(90 + angle)

# otherwise, just take the inverse of the angle to make
# it positive
else:
    angle = -angle
angle = -angle

# rotate the image to deskew it
(h, w) = orig.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(orig, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# rotated = imutils.rotate_bound(img, angle=angle)
# rotated = imutils.rotate_bound(img, angle=-angle)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
cv2.imshow("Input", img)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)


gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 60, 255, 1)
blur = cv2.medianBlur(thresh, 1)

print(pytesseract.image_to_string(gray, lang='ara'))

cv2.imshow('ImageWindow1', gray)
cv2.imshow('ImageWindow2', thresh)
cv2.imshow('ImageWindow3', blur)

cv2.waitKey(0)
