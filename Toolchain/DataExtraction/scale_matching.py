# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2


def submatrix_video(matrix, p1=(1000, 190), p2=(1200, 260)):
    return matrix[p1[1]:p2[1], p1[0]:p2[0]], p1[0], p1[1]


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-m", "--mask", required=True, help="Path to template mask")
ap.add_argument("-i", "--images", required=True,
                help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
                help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)

mask = cv2.imread(args["mask"])

(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

# loop over the images to find the template in
scales = np.linspace(.5, 1.75, 21)[::-1]
print(scales)

for imagePath in glob.glob(args["images"] + "/*.png"):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    frame_h = image.shape[0]
    frame_w = image.shape[1]

    # shape[0] = height, shape[1] = width
    # if frame_h == 720:
    #   image, _, _ = submatrix_video(image, p1=(int(frame_w/3)*2, 0), p2=(int(frame_w), int(frame_h/2.5)))

    # elif frame_h == 1080:
    image, _, _ = submatrix_video(image, p1=(int(frame_w / 4) * 3, 0), p2=(int(frame_w), int(frame_h / 2.5)))

    # else:
    #   print("Y? T.T how is u {:f} high".format(frame_h))

    '''if image.shape[1] < template.shape[1] or image.shape[0] < template.shape[0]:
        #print("image.shape[1]: {:d} < template.shape[1]: {:d}; image.shape[0]: {:d} < template.shape[0]: {:d}; ".format(image.shape[1],template.shape[1],image.shape[0],template.shape[0]))
        print("image to small: {:s}".format(imagePath))
        continue'''

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(gray, (2, 2))
    detected_edges = cv2.Canny(img_blur, 70, 70 * 3, 3)
    found = None

    # loop over the scales of the image
    if frame_h == 720:
        scales = np.linspace(1.40, 1.65, 21)[::-1]
    else:
        scales = np.linspace(.95, 1.35, 21)[::-1]
        # scales = np.arange(.5, 1.40, .05)[::-1]

    print("frame_h:\t{:d}".format(frame_h))
    print(scales)
    for scale in scales:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        # resized = cv2.resize(gray, None, None, scale, scale, cv2.INTER_CUBIC)
        # resized = cv2.resize(img_blur, None, None, scale, scale, cv2.INTER_CUBIC)
        resized = cv2.resize(detected_edges, None, None, scale, scale, cv2.INTER_CUBIC)
        # resized = imutils.resize(detected_edges, width=int(detected_edges.shape[1] * scale))
        # resized = imutils.resize(gray, width=int(detected_edges.shape[1] * scale))
        r = detected_edges.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 75, 75 * 3, 3)  # cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF, mask)
        # result = cv2.matchTemplate(edged, template, cv2.TM_SQDIFF_NORMED, mask)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if args.get("visualize", False):
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 1)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            # if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, scale)
        # print(found)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    print(imagePath)
    if found == None:
        print("Nothing found!")
    else:
        (maxVal, maxLoc, r, scale) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        # draw a bounding box around the detected result and display the image
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        print("scale:\t{:1.4f}\tratio:\t{:f}\tmaxVal:\t{:f}\tframe h:\t{:d}".format(scale, r, maxVal, frame_h))
        cv2.resizeWindow('Image', 1000, 1000)
        cv2.imshow("Image", image)
        cv2.imwrite(imagePath[0:-4] + "_found.png", image)
        cv2.waitKey(0)
