from __future__ import print_function
import cv2 as cv
import argparse

import numpy as np

max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (2,2))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    if val>=90 and val <= 99:
        cv.imwrite(args.input[0:args.input.rfind(".")]+'_val{:3d}_.png'.format(val), detected_edges)
    cv.imshow(window_name, detected_edges)

parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='../data/fruits.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()

img = cv.imread(args.input, cv.IMREAD_UNCHANGED)

#convert img to grey
img_grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv.threshold(img_grey, thresh, 255, cv.THRESH_BINARY)
#find contours
cont_image, contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv.drawContours(img_contours, contours, -1, (0,255,0), cv.CHAIN_APPROX_NONE)
#save image
cv.imwrite('/home/itec-linux/Pictures/edges/contours_sidebar.png',img_contours)

cv.namedWindow("conture")
cv.imshow("conture", img_contours)
cv.waitKey()