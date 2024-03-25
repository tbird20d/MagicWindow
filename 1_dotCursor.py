#!/usr/bin/python3
import cv2
import numpy as np

# prevx = 0
# prevy = 0

#img = cv2.imread('assets/blackScreen.jpg', -1)
#img = cv2.resize(img, (900, 600))

img = np.zeros(shape=(600, 900, 3), dtype=np.uint8)
blank_img = img.copy()

cv2.imshow('image', img)


def mouse_move(event, x, y, flags, param):

    global blank_img
    global img
    # global prevx
    # global prevy

    # if prevx != 0 or prevy != 0:
    #     cv2.circle(img, (prevx, prevy), 5, (0, 0, 0), -1)
    # else:
    #     img = blank_img

    # blank_img = img.copy()

    if event == cv2.EVENT_MOUSEMOVE:
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.imshow('image', img)
        # prevx = x
        # prevy = y

    img = blank_img


cv2.setMouseCallback('image', mouse_move)

cv2.waitKey(0)
cv2.destroyAllWindows()
