import numpy as np
import cv2 as cv
import glob
import argparse


def submatrix_video(matrix, p1=(1000, 190), p2=(1200, 260)):
    return matrix[p1[1]:p2[1], p1[0]:p2[0]], p1[0], p1[1]


def locate_white_line(image, image_orig_h):

    # HLS
    lower_white = np.array([0, 190, 0])
    upper_white = np.array([255, 255, 255])

    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS_FULL)

    # Threshold the HLS image to get only white colors
    image = cv.inRange(hls, lower_white, upper_white)
    cv.imshow("black/white", image)

    if image_orig_h == 720:
        thresh_low = 180
        thresh_high = 210
    else:
        thresh_low = 280
        thresh_high = 310
    res = []
    line_start = None
    line_count = 0
    count = 0
    # go through columns of the image
    for x in range(0, image.shape[1]):
        # go through rows of the column
        for y in range(image.shape[0]):
            # check if pixel is white
            if image[y][x] == 255:
                count += 1
                line_count += 1
                if line_count == 1:
                    line_start = (x, y)
            else:
                if line_count < 30:
                    line_start = None
                    line_count = 0

        if thresh_low <= count <= thresh_high:
            res.append((line_start, count))
        count = 0
        line_start = None
        line_count = 0

    return res


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="Path to images where template will be matched")
    ap.add_argument("-v", "--visualize",
                    help="Flag indicating whether or not to visualize each iteration")
    args = vars(ap.parse_args())

    for imagePath in glob.glob(args["images"] + "/*.png"):
        image = cv.imread(imagePath)
        frame_h = image.shape[0]
        frame_w = image.shape[1]


        region_, x1_, y1_ = submatrix_video(image, p1=(int(frame_w / 4) * 3, 0), p2=(int(frame_w), int(frame_h / 2.5)))

        #b_w_image = black_white_image(region_)
        # shape[0] = height, shape[1] = width

        location = locate_white_line(image, frame_h)
        print(location)

        cv.namedWindow(imagePath.split("/")[-1], cv.WINDOW_KEEPRATIO)
        cv.imshow(imagePath.split("/")[-1], image)
        if frame_h == 720:
            cv.resizeWindow(imagePath.split("/")[-1], (600, 400))
        else:
            cv.resizeWindow(imagePath.split("/")[-1], (800, 500))
        cv.waitKey(0)
        cv.destroyAllWindows()

