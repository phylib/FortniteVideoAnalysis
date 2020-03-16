import cv2
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: {} input-file.png".format(sys.argv[0]))

threshold = 1200

path = sys.argv[1]
img = cv2.imread(path, 0)
img = cv2.resize(img, (2000, 2000))
# ret, thresh = cv2.threshold(img, 255 * 0.35, 255, cv2.THRESH_BINARY)
# blur = cv2.GaussianBlur(img, (63, 63), 0)
# thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)

# get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:, cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= threshold:  # keep
        result[labels == i + 1] = 255
result = cv2.GaussianBlur(result, (63, 63), 0)
result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)[1]

all_pixels = 2000 * 2000
nonzero_pixels = np.count_nonzero(result)
print("Percentage of hotspot pixels: {:.2f}".format(float(nonzero_pixels) / all_pixels))
cv2.imwrite(path.replace(".png", "_edited.png"), result)

# thresh = cv2.resize(thresh, (400, 400))
# img = cv2.resize(img, (400, 400))
# cv2.imshow('original', img)
# cv2.imshow('output', thresh)
# time.sleep(10)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
