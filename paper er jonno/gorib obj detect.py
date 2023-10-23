import cv2
import numpy as np

farm_img = cv2.imread('farm.png', cv2.IMREAD_UNCHANGED)
needle_img = cv2.imread('needle.png', cv2.IMREAD_UNCHANGED)

# cv2.imshow('Farm', farm_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Needle', needle_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

result = cv2.matchTemplate(farm_img, needle_img, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

w = needle_img.shape[1]
h = needle_img.shape[0]

cv2.rectangle(farm_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)

threshold = 0.60

yloc, xloc = np.where(result >= threshold)

for (x, y) in zip(xloc, yloc):
    cv2.rectangle(farm_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('Farm', farm_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# What is a rectangle?
# x, y, w, h
rectangles = []
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])

rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
