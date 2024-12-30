import cv2
import numpy as np

COLOR = (255, 255, 0)
TARGET_SIZE = 25000
TARGET_NAME = 'card'

img = cv2.imread('../resources/cards.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, otsu = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

img_copy = img.copy()

idx = 1
for contour in contours:
    x, y, width, height = cv2.boundingRect(contour)
    if cv2.contourArea(contour) > TARGET_SIZE:
        cv2.rectangle(img, (x, y), (x+width, y+height), COLOR)

        target = img_copy[y:y+height, x:x+width]
        cv2.imwrite(f'../results/{TARGET_NAME}{idx}.jpg', target)
        cv2.imshow('img', img)
        idx += 1

cv2.waitKey(0)
cv2.destroyAllWindows()