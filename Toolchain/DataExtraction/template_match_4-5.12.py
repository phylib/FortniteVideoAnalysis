import cv2 as cv
import numpy as np
import time
import math
import re
import pyocr.builders
from PIL import Image
import argparse
import os.path


def clamp(minimum, maximum, val):
    return max(minimum, min(val, maximum))


def submatrix(res, p, n):
    x1 = clamp(0, len(res) - 1, p[0] - n)
    x2 = clamp(0, len(res) - 1, p[0] + n)
    y1 = clamp(0, len(res) - 1, p[1] - n)
    y2 = clamp(0, len(res) - 1, p[1] + n)
    return res[y1:y2, x1:x2], x1, y1


### Crops the map frame form the video
def submatrix_video(matrix, p1=(1000, 190), p2=(1200, 260)):
    return matrix[p1[1]:p2[1], p1[0]:p2[0]], p1[0], p1[1]


def phase_find_best_match(region, phase_icons, mask_icon):
    result = []
    max_values = []
    max_locations = []

    for icon_idx in range(0, len(phase_icons)):
        result.append(cv.matchTemplate(region, phase_icons[icon_idx], cv.TM_CCORR_NORMED, mask=mask_icon))
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result[-1])
        max_values.append(max_val)
        max_locations.append(max_loc)

    # Phases index
    # 0 - jump
    # 1 - time till contract
    # 2 - contracting
    # phase_icons are checked and added in Phases index order to the max_values
    max_idx = np.argmax(max_values)
    return max_idx, max_values[max_idx], max_locations[max_idx]


def read_num_next_to_icon(frameV, frame_h, icon, icon_mask, lastPosition_icon, tool, builder):
    # crop region for searching according to video size
    if frame_h == 720:
        res = cv.matchTemplate(
            frameV[lastPosition_icon[1]:lastPosition_icon[1] + 25, lastPosition_icon[0]:1215],
            icon,
            cv.TM_CCORR_NORMED, mask=icon_mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        text_img = frameV[max_loc[1] + lastPosition_icon[1]:max_loc[1] + lastPosition_icon[1] + 20,
                   max_loc[0] + lastPosition_icon[0] + 24:max_loc[0] + lastPosition_icon[0] + 45]
    else:
        res = cv.matchTemplate(
            frameV[lastPosition_icon[1]:lastPosition_icon[1] + 25, lastPosition_icon[0]:1810],
            icon,
            cv.TM_CCORR_NORMED, mask=icon_mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        text_img = frameV[max_loc[1] + lastPosition_icon[1]:max_loc[1] + lastPosition_icon[1] + 20,
                   max_loc[0] + lastPosition_icon[0] + 24:max_loc[0] + lastPosition_icon[0] + 45]

    # show cropped image & place found icon position on the video frame
    if visualization_read_num_next_to_icon:
        cv.imshow("text search region", region)
        cv.imshow("text_img", text_img)
        cv.rectangle(frameV, (lastPosition_icon[0] + max_loc[0], lastPosition_icon[1] + max_loc[1]),
                     (lastPosition_icon[0] + max_loc[0] + icon.shape[1],
                      lastPosition_icon[1] + max_loc[1] + icon.shape[0]),
                     255, 1)
        # cv.imshow("funktion_icon", frameV)
        cv.waitKey(1)
    # read the text next to the icon
    text_img = cv.cvtColor(text_img, cv.COLOR_BGR2GRAY)
    text_img = cv.bilateralFilter(text_img, 5, 100, 100)
    text_img = cv.equalizeHist(text_img)
    res_img = cv.threshold(text_img, 200, 255, cv.THRESH_BINARY_INV)[1]
    pil_im = Image.fromarray(res_img)

    # Setting in tesseract for digits recognition(-> changing the page segmentation mode)
    # builder.tesseract_layout = 3
    # builder.tesseract_flags = ['-psm', '3']  # If tool = libtesseract

    builder.tesseract_configs = ['--oem', '0', '--psm', '3', 'digits', '-c', 'tessedit_char_whitelist=0123456789']

    # print(builder.tesseract_configs)

    number = tool.image_to_string(
        pil_im,
        lang=lang,
        builder=builder
    ).replace('.', ' ').replace('-', ' ').strip()
    # if no digit recognised or wrong one try setting reading for 1 digit
    if number == "" or len(number) > 2:
        # Setting in tesseract for single digit recognition-> changing the page segmentation mode
        builder.tesseract_configs = ['--oem', '0', '--psm', '10', 'digits', '-c', 'tessedit_char_whitelist=0123456789']

        # print(builder.tesseract_configs)

        res_img = cv.threshold(text_img, 225, 255, cv.THRESH_BINARY_INV)[1]
        pil_im = Image.fromarray(res_img)

        number = tool.image_to_string(
            pil_im,
            lang=lang,
            builder=builder
        ).replace('.', ' ').replace('-', ' ').strip()

    if output_read_num_next_to_icon:
        print("\t\t{:s}".format(number))

    return number, res_img


# default values for player_count
def count_pattern_and_validity_check(last_known_valid_numbers, last_recognised_count, varianz):  #
    varianz_down_normal = varianz[0]
    varianz_up_normal = varianz[1]
    varianz_down_big = varianz[2]
    varianz_up_big = varianz[3]

    newNr = last_recognised_count[-1]

    if output_read_num_next_to_icon:
        print('before:', last_known_valid_numbers, last_recognised_count)

    if newNr != -666:
        # check if new number is plausible with last valid number
        if last_known_valid_numbers[-1] + varianz_up_normal >= newNr >= last_known_valid_numbers[-1] \
                - varianz_down_normal:
            last_known_valid_numbers.append(newNr)
            last_known_valid_numbers = last_known_valid_numbers[1:]
        # number not plausible so check if last recog. numbers are all the same as new number & set it as valid number
        elif newNr == last_recognised_count[-2] == last_recognised_count[-3]:  # \
            # and int(last_known_valid_numbers[-1]) - varianz_down_big <= int(newNr) \
            # <= int(last_known_valid_numbers[-1]) + varianz_up_big:
            last_known_valid_numbers.append(newNr)
            last_known_valid_numbers = last_known_valid_numbers[1:]

    else:
        # check if last 2 recog. nr are constant and stick to that
        if last_recognised_count[-2] == last_recognised_count[-3] != -666:
            last_known_valid_numbers.append(last_recognised_count[-2])
            last_known_valid_numbers = last_known_valid_numbers[1:]

    if output_read_num_next_to_icon:
        print('after:', last_known_valid_numbers, last_recognised_count)

    return last_known_valid_numbers


# Return Scale for frame (frame_h = original frame height)
def getScale(image, frame_h, frame_w, template, template_mask, dirName=None, filename=None, color  = (0, 0, 255)):
    # load the image image, convert it to grayscale, and detect edges
    # template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # template = cv.Canny(template, 50, 200)

    (tH, tW) = template.shape[:2]
    ratio = 3
    kernel_size = 3

    # shape[0] = height, shape[1] = width
    offsetX = int(frame_w / 4) * 3
    offsetY = int(frame_h / 2.5)
    image_crop, _, _ = submatrix_video(image, p1=(offsetX, 0), p2=(int(frame_w), offsetY))

    gray = cv.cvtColor(image_crop, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(gray, (3, 3))
    detected_edges = cv.Canny(img_blur, 70, 70 * ratio, kernel_size)  # 70,70
    found = None

    # set scales fitting for the different formats
    ###if frame_h == 720:
    ###    scales = np.linspace(1.40, 1.60, 21)[::-1]
    ###else:
    scales = np.linspace(.90, 1.60, 11)[::-1]

    # loop over the scales of the image
    # print(scales)
    for scale in scales:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = cv.resize(detected_edges, None, None, scale, scale, cv.INTER_CUBIC)
        # resized = imutils.resize(detected_edges, width=int(detected_edges.shape[1] * scale))
        # resized = imutils.resize(gray, width=int(detected_edges.shape[1] * scale))
        r = detected_edges.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv.Canny(resized, 75, 75 * 3, 3)  # cv.Canny(resized, 50, 200)
        result = cv.matchTemplate(edged, template, cv.TM_CCOEFF_NORMED, template_mask)
        # result = cv.matchTemplate(edged, template, cv.TM_SQDIFF_NORMED, template_mask)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        # (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if True:  # visualization_map:
            # draw a bounding box around the detected region
            # clone = np.dstack([edged, edged, edged])
            # cv.rectangle(clone,
            #             (maxLoc[0], maxLoc[1]),
            #             (maxLoc[0] + tW, maxLoc[1] + tH),
            #            (0, 0, 255), 1)
            '''cv.imshow("Visualize", clone)'''
            # cv.waitKey(1)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r, scale)
            # print(found)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    # print(imagePath)
    (maxVal, maxLoc, r, scale) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    if not (dirName is None and time is None):
        outpath = dirName + "/sclaes_out_" + filename + "_{:.0f}.csv".format(frame_h)
        # print("writing to scales csv("+outpath+")")

        if not os.path.exists(outpath):
            outfile = open(outpath, "a")
            outfile.write(
                "maxVal;Xpos;Ypos;Xsize;Ysize;scale;ratio\n")
            outfile.close()

        outfile = open(outpath, "a")
        outfile.write(
            "{:1.4f};{:1.4f};{:1.4f};{:1.4f};{:1.4f};{:1.4f};{:1.4f}\n".format(maxVal, startX + offsetX, startY,
                                                                template.shape[1] * scale, template.shape[0] * scale,
                                                                scale, r))
        outfile.close()

    # draw a bounding box around the detected result and display the image
    img_display = image.copy()

    cv.rectangle(img_display, (startX + offsetX, startY), (endX+ offsetX, endY), color, 2)
    print("maxVal: {:f}\t frame h: {:f}\tscale: {:1.4f}\tratio: {:f}".format(maxVal, frame_h, scale, r))
    #cv.resizeWindow('Image', 1000, 1000)
    cv.imshow("Scale + location result", img_display)
    # cv.imwrite(imagePath[0:-4]+"_found.png", image)
    cv.waitKey(0)

    return maxVal, (startX + offsetX, startY), r, scale


def locate_map_scale(image):

    # shape[0] = height, shape[1] = width
    offsetX = int(image.shape[1] / 4) * 3
    offsetY = int(image.shape[0] / 2.5)
    image_crop, _, _ = submatrix_video(image, p1=(offsetX, 0), p2=(int(image.shape[1]), offsetY))

    # HLS
    lower_white = np.array([0, 200, 0]) #190
    upper_white = np.array([255, 255, 255])

    hls = cv.cvtColor(image_crop, cv.COLOR_BGR2HLS_FULL)

    # TODO Histogramm check if too much white/light grey
    # https://answers.opencv.org/question/198738/getting-the-hsl-conversion-lightness-average-value-from-an-image/
    image_crop_L = cv.extractChannel(hls, 1)

    histSize = 255
    #      cv.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    hist = cv.calcHist(image_crop_L, [1], None, [histSize], (0, 256))

    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))
    histImage = np.zeros((hist_h, hist_w), dtype=np.uint8)

    #cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

    for i in range(1, histSize):
        #cv.line(histImage, (bin_w * (i - 1), hist_h - int(round(hist[i - 1]))),
                #(bin_w * (i), hist_h - int(round(hist[i]))),
        print(hist[i - 1])
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(hist[i - 1])),
                (bin_w * (i), hist_h - int(hist[i])),
                (255, 0, 0), thickness=2)

    if image.shape[0] == 1080:
        padding = 5
    else:
        padding = 3

    # Threshold the HLS image to get only white colors
    image_crop = cv.inRange(hls, lower_white, upper_white)

    if image.shape[0] == 720:
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
    for x in range(0, image_crop.shape[1]):
        # go through rows of the column
        for y in range(image_crop.shape[0]):
            # check if pixel is white
            if image_crop[y][x] == 255:
                count += 1
                line_count += 1
                if line_count == 1:
                    line_start = (x, y)
            else:
                if line_count < 30:
                    line_start = None
                    line_count = 0

        if thresh_low <= count <= thresh_high:
            if image_crop.shape[1] - count - padding > line_start[0] and line_start[1] > 10:
                res.append((line_start, count, line_count))
        count = 0
        line_start = None
        line_count = 0

    img_display = image.copy()

    if len(res) > 0:
        cv.rectangle(img_display,
                     (res[0][0][0] + offsetX + padding, res[0][0][1]),
                     (res[0][0][0] + offsetX + res[0][1] + padding, res[0][0][1] + res[0][1]),
                     (0, 0, 255), 2)
    #print("maxVal: {:f}\t frame h: {:f}\tscale: {:1.4f}\tratio: {:f}".format(maxVal, frame_h, scale, r))
    #print(res)
    #cv.resizeWindow('Image', 1000, 1000)
    cv.imshow("Scale + location result", img_display)
    # cv.imwrite(imagePath[0:-4]+"_found.png", image)
    cv.imshow('cropped b/w', image_crop)
    cv.imshow('cropped grey', image_crop_L)
    cv.imshow('calcHist Demo', histImage)
    cv.waitKey(1)



    return (res[0][0][0] + offsetX + padding, res[0][0][1]), \
           (res[0][0][0] + offsetX + res[0][1] + padding, res[0][0][1] + res[0][1]), \
            scale


# File 1: all videos // won: -1 = abborted round; 0 = Player died; 1 = Player won
def write_all(outfile, video, roundId, starttime, endtime, won, place, kills):
    outfile = open(outfile.name, "a")
    outfile.write(
        "{:s},{:d},{:s},{:s},{:d},{:d},{:d},\n".format(video, roundId, starttime, endtime, won, place, kills))
    outfile.close()
    return None


# File 2: per Video_RoundId:
def write_round(outfile, phase, x_pos, y_pos, distance, video_time, place, kills):
    outfile = open(outfile.name, "a")
    outfile.write(
        "{:1d},{:4d},{:4d},{:6.3f},{:s},{:d},{:d}\n".format(phase, x_pos, y_pos, distance, video_time, place, kills))
    outfile.close()
    return None


### visualization configure here ...

visualization_map = True
visualization_phase = True
visualization_deathORwin = True
visualization_read_num_next_to_icon = True
# visualization_read_num_next_to_icon = False
output_read_num_next_to_icon = True
# output_read_num_next_to_icon = False
visualization_general_info = True

### variable configure here...
# skip ahead some frames
jump_ahead_min = 0  # 7
jump_ahead_sec = 0  # 50

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to parse fortnite Videos.')
    parser.add_argument("-i", "--input", help="Video file to process")
    # parser.add_argument("-m", "--metaoutput", help="Output file for round meta information")
    # parser.add_argument("-r", "--roundoutput", help="Output file for detailed information off a round")
    parser.add_argument("-v", "--verbose", help="how many output to print, default, less", type=bool, default=False)
    parser.add_argument("-o", "--outpath", help="Path for output")
    parser.add_argument("-t", "--testrun", help="debug test run mode for single videos", type=bool, default=False)

    args = parser.parse_args()
    filename = args.input
    # output_all = args.metaoutput
    # output_round = args.roundoutput
    verbose = args.verbose
    output_path = args.outpath
    testrun = args.testrun

    if testrun:
        output_timestamp = time.strftime("%d-%m-%y_%H.%M.%S", time.gmtime())
    else:
        output_timestamp = time.strftime("%d-%m-%y", time.gmtime())

    dirName = output_path + '/fn-dataextraction'
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    dirName = dirName + '/fn-dataextraction_' + output_timestamp
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")

    output_all = dirName + '/match-overview.csv'
    output_round = dirName + '/round-'

    if not verbose:
        visualization_map = False
        visualization_phase = False
        visualization_deathORwin = False
        visualization_read_num_next_to_icon = False
        output_read_num_next_to_icon = False
        visualization_general_info = False

    # setting up map for position tracing
    img_map = cv.imread('Chapter_2_Season_1_Minimap_2000x2000_edited.png', cv.IMREAD_GRAYSCALE)
    img_map_mask = cv.imread('map_mask_ch2_2000x2000.png', cv.IMREAD_GRAYSCALE)
    vis_color = cv.imread('Chapter_2_Season_1_Minimap_2000x2000_edited.png', cv.IMREAD_COLOR)
    # vis_color = cv.resize(vis_color, None, None, 1.39, 1.39, cv.INTER_CUBIC) # Scale ok
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_map = clahe.apply(img_map)

    heatmap = np.zeros(img_map_mask.shape)

    lastPositions = []

    start_time = time.time()

    # check if video is form UseCaseStudy -> different cropping settings
    '''if filename.__contains__("Fn"):
        FnUseCaseStudy = True
    else:
        FnUseCaseStudy = False'''

    cap = cv.VideoCapture(filename)

    frameRate = math.floor(cap.get(cv.CAP_PROP_FPS))
    ### Set variable for video size (720p or 1080p)to check while analysing
    video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    ret = True

    ### skip ahead some frames
    for c in range(0, math.floor(((jump_ahead_min * 60) + jump_ahead_sec) * frameRate)):
        ret, frameV = cap.read()

    ######## finde Phase pt1
    # set up img to text tool
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    if visualization_read_num_next_to_icon or output_read_num_next_to_icon or visualization_general_info:
        print("Will use tool '%s'" % (tool.get_name()))

    lang = 'eng'
    builder = pyocr.tesseract.DigitBuilder()
    # builder = pyocr.builders.TextBuilder()
    # builder = pyocr.builders.LineBoxBuilder()

    # Phases index
    # 0 - jump
    # 1 - time till contract
    # 2 - contracting

    # images for phase recognition
    mask_icon = cv.imread("mask_icon.png")
    person_icon_mask = cv.imread("person_icon_mask.png")

    person_icon = cv.imread("person_icon.png")

    kills_icon = cv.imread("kills_icon.png")

    phase_icons = [cv.imread("jump_icon.png", cv.IMREAD_COLOR),
                      cv.imread("time_icon_.png", cv.IMREAD_COLOR),
                      cv.imread("storm_icon.png", cv.IMREAD_COLOR)]

    # image for check if jump is really jump
    jump_check_icon = cv.imread("jump_check_icon.png", cv.IMREAD_COLOR)

    # last positions of icon (x,y)
    lastPosition_icon = (0, 0)
    # avg last positions of icon (x, y, count)
    avgLastPosition_icon = (0, 0, 0)
    # samePositionCount = 0
    current_phase = -1

    win_mask = cv.imread("win_1080.png")
    #death_mask = cv.imread("death_mask_720_all.png")

    # new round starting?
    waiting4newround = True
    roundId = 0

    if not os.path.exists(output_all):
        outfile_all = open(output_all, "a")
        outfile_all.write("video,roundId,starttime,endtime,won,place,kills\n")
        outfile_all.close()
    else:
        outfile_all = open(output_all, "r")
        outfile_all.close()

    frameTimeStamp = ""
    starttime = ""

    ######## finde Phase pt1 end

    # list of last 3 valid player count numbers
    last_known_valid_player_count = [100] * 3
    # list of last 3 frames recognised player count
    last_recognised_player_count = [100] * 3
    # (varianz down, varianz up)
    playerCountVarianz = (4, 3, 35, 35)

    # kills
    # list of last 3 valid kills numbers
    last_known_valid_kill_count = [0] * 3
    # list of last 3 frames recognised kills
    last_recognised_kill_count = [0] * 3
    # (varianz down, varianz up)
    playerKillVarianz = (1, 1, 5, 5)

    # pattern for regex check of image to text output
    pattern_player_count = re.compile('^[1-9][0-9]?$')
    pattern_player_kills = re.compile('^[1-9]?[0-9]$')

    scale = None
    # load the image image, convert it to grayscale, and detect edges
    scale_template_person = cv.imread("scale_edges_person_icon.png")
    scale_template_person = cv.cvtColor(scale_template_person, cv.COLOR_BGR2GRAY)
    scale_template_jump = cv.imread("scale_edges_jump_icon.png")
    scale_template_jump = cv.cvtColor(scale_template_jump, cv.COLOR_BGR2GRAY)
    scale_template_map = cv.imread("scale_edges_map_icon.png")
    scale_template_map = cv.cvtColor(scale_template_map, cv.COLOR_BGR2GRAY)
    scale_template_sidebar = cv.imread("sidebar_map.png")
    scale_template_sidebar = cv.cvtColor(scale_template_sidebar, cv.COLOR_BGR2GRAY)
    # scaled mask
    scale_mask = None  # cv.imread("scale_edges_mask.png")
    scale_mask_sidebar = None  # cv.imread("sidebar_map_mask.png")

    scales = []
    scale_found = False
    scale_final = None
    scale_buffered_frames = []

    # go through frame by frame
    while cap.isOpened() and ret:
        # get next frame
        ret, frameV = cap.read()
        if not ret:
            break

        ### Set variables according to video size (720p or 1080p)
        frameTime = math.floor(cap.get(cv.CAP_PROP_POS_MSEC) / 1000)
        video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
        video_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float

        # current frame number
        frameId = cap.get(cv.CAP_PROP_POS_FRAMES)
        frameTimeStamp = "{:02d}:{:02d}:{:02d}".format(math.floor(frameTime / 60 / 60) % 60,
                                                       math.floor((frameTime / 60) % 60),
                                                       # math.floor(frameTime / 60),
                                                       math.floor(frameTime) % 60)  # in seconds
        # enter every second of the video
        if frameId % frameRate == 0:  # and frameTimeStamp % 1000 == 0:
            if visualization_general_info:
                print(
                    "\nframeid: %i out of %i\tTimestamp: %s\t file: %s" % (
                        frameId, cap.get(cv.CAP_PROP_FRAME_COUNT), frameTimeStamp, filename.split("/")[-2:]))
            ######## finde Phase pt2

            if not scale_found:
                if True:  # len(scales) < 20:
                    '''#scales.append((getScale(frameV, video_height, video_width, scale_template, scale_mask, dirName,
                    #                        filename.split("/")[-2] + "_" + output_timestamp)))
                    #getScale(frameV, video_height, video_width, scale_template_person, scale_mask, dirName,
                    #         filename.split("/")[-2] + "_" + output_timestamp, (0, 0, 255))
                    #getScale(frameV, video_height, video_width, scale_template_jump, scale_mask, dirName,
                    #         filename.split("/")[-2] + "_" + output_timestamp, (0, 255, 0))
                    getScale(frameV, video_height, video_width, scale_template_map, scale_mask, dirName,
                             filename.split("/")[-2] + "_" + output_timestamp, (255, 0, 0))
                    getScale(frameV, video_height, video_width, scale_template_sidebar, scale_mask_sidebar, dirName,
                             filename.split("/")[-2] + "_" + output_timestamp, (0, 0, 255))
                    # scale_buffered_frames.append(frameV)
                    # if len(scales) > 5:
                    #    scale_buffered_frames[1:]
                    # TODO find maxVal needed for positive scale results (maybe combine it with a jumpphase check?)
                    # print(scales)'''
                    print(locate_map_scale(frameV))

            # cut search region and find out phase
                region, x1, y1 = submatrix_video(frameV, (1010, 220), (1100, 250))  # 1100        1170
                current_phase, max_val, max_loc = phase_find_best_match(region, phase_icons, mask_icon)

            else:
                # TODO process through the scales buffered frames before continuing further frames
                # TODO while len(scale_buffered_frames) > 0: {process scale_buffered_frames[0]; scale_buffered_frames[1:]}
                # TODO reconsider where to call  "ret, frameV = cap.read()"

                region, x1, y1 = submatrix_video(frameV, (1015, 200), (1080, 260))  # 1100        1200
                current_phase, max_val, max_loc = phase_find_best_match(region, phase_icons, mask_icon)


                if visualization_general_info:
                    # if no matching phase was found set phase to -1
                    print("Max_val: {:f} currentphase found: {:d}".format(max_val, current_phase))

                if (current_phase == 0 and max_val < .91) or max_val < .966:  # .96 --> .954 --> .966
                    if visualization_general_info:
                        print("current_phase set to -1".format(max_val))
                    current_phase = -1

                lastPosition_icon = (max_loc[0] + x1, max_loc[1] + y1)
                top_left = (max_loc[0], max_loc[1])

                if visualization_phase:

                    cv.rectangle(frameV, (top_left[0] + x1, top_left[1] + y1),
                                 (top_left[0] + x1 + mask_icon.shape[1],
                                  top_left[1] + y1 + mask_icon.shape[0]),
                                 255, 3)
                    # cv.imshow("video", frameV)
                    cv.imshow("region_Phase", region)
                    cv.waitKey(3)

                # if in a round and no phase was found or icon position changed: check if player died/won
                if not waiting4newround:
                    if ((current_phase <= 0)
                            or ((lastPosition_icon[0] <= avgLastPosition_icon[0] - 2 or
                                 lastPosition_icon[0] >= avgLastPosition_icon[0] + 2)
                                or (lastPosition_icon[1] >= avgLastPosition_icon[1] + 2 or
                                    lastPosition_icon[1] <= avgLastPosition_icon[1] - 2))):
                        # HLS
                        lower_white = np.array([0, 230, 0])
                        upper_white = np.array([255, 255, 255])

                        # Check if player died
                        region_, x1_, y1_ = submatrix_video(frameV, (450, 95), (450 + 360, 95 + 60))
                        hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                        # Threshold the HLS image to get only white colors
                        mask = cv.inRange(hls, lower_white, upper_white)
                        # Bitwise-AND mask and original image
                        res = cv.bitwise_and(region_, region_, mask=mask)

                        death, max_val_, max_loc_ = phase_find_best_match(res, [death_mask], None)

                        if visualization_deathORwin:
                            # cv.imshow("deathframecrop", res)
                            print("Max_val_ death: {:f}".format(max_val_))
                        # if max_val_ of player death high enough write that player died
                        if max_val_ >= .90:

                            write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, 0,
                                      last_known_valid_player_count[-1], last_known_valid_kill_count[-1])
                            waiting4newround = True
                            if visualization_deathORwin:
                                print(
                                    "|#############################################DEATH####################################|")

                        # check if player won
                        else:
                            region_, x1_, y1_ = submatrix_video(frameV, (325, 120), (325 + 535, 120 + 110))
                            hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                            # Threshold the HLS image to get only white colors
                            mask = cv.inRange(hls, lower_white, upper_white)
                            # Bitwise-AND mask and original image
                            res = cv.bitwise_and(region_, region_, mask=mask)

                            win, max_val_, max_loc_ = phase_find_best_match(res, [win_mask], None)

                            if visualization_deathORwin:
                                # cv.imshow("winframecrop", res)
                                # cv.imwrite("WinCrop_{:06.0f}.png".format(frameId), res)
                                print("Max_val_ win: {:f}".format(max_val_))
                            # if max_val_ of player won high enough write that player won
                            if max_val_ >= .90:
                                '''outfile.write("\n\nWIN\t{:s}\n\n".format(frameTimeStamp))
                                outfile.write(
                                    "Win with: {:06.0f}\t{:s}\t{:1d}\t{:4d}\t{:4d}\t{:1d}\t{:1.5f}\t{:6.3f}\t{:s}\t{:s}\n".format(
                                        frameId,
                                        frameTimeStamp,
                                        current_phase,
                                        int(
                                            top_left[
                                                0] + w / 2),
                                        int(
                                            top_left[
                                                1] + h / 2),
                                        omitFrame,
                                        max_val,
                                        distance,
                                        str(int(last_known_valid_player_count[-1]) - 1),
                                        last_known_valid_kill_count[-1]))'''
                                # if only 2 players are left and player wins -> one more kill to the count
                                corrected_win_kills = last_known_valid_kill_count[-1]
                                corrected_player_count = last_known_valid_player_count[-1]

                                if last_known_valid_player_count[-1] == 2:
                                    corrected_win_kills = corrected_win_kills + 1

                                # if win not imideatly recognissed & updated count recognised -> +1 to playercount to correct output
                                elif last_known_valid_player_count == 1:
                                    corrected_player_count += 1

                                write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, 1,
                                          corrected_player_count - 1,
                                          corrected_win_kills)
                                waiting4newround = True
                                if visualization_deathORwin:
                                    print(
                                        "|#############################################WIN####################################|")

                # Update avg last pos after useing it in check
                if not waiting4newround and current_phase != -1:
                    if lastPosition_icon[0] > avgLastPosition_icon[0] and last_recognised_player_count[-1] > 10 or \
                            lastPosition_icon[0] < avgLastPosition_icon[0] and last_known_valid_kill_count[-1] > 9:
                        avgLastPosition_icon = (0, 0, 0)
                    avgLastPosition_icon = ((avgLastPosition_icon[0] * avgLastPosition_icon[2] + lastPosition_icon[0]) /
                                            (avgLastPosition_icon[2] + 1),
                                            (avgLastPosition_icon[1] * avgLastPosition_icon[2] + lastPosition_icon[1]) /
                                            (avgLastPosition_icon[2] + 1),
                                            avgLastPosition_icon[2] + 1)

                if visualization_general_info or visualization_deathORwin or visualization_phase:
                    print("AvgLast: ({:.3f}x{:.3f},  {:.0f})\tLast: ({:.3f}x{:.3f})".format(avgLastPosition_icon[0],
                                                                                            avgLastPosition_icon[1],
                                                                                            avgLastPosition_icon[2],
                                                                                            lastPosition_icon[0],
                                                                                            lastPosition_icon[1]))

                if current_phase == 0:
                    # HLS
                    lower_white = np.array([0, 180, 0])
                    upper_white = np.array([255, 255, 255])
                    hls = cv.cvtColor(region, cv.COLOR_BGR2HLS_FULL)
                    # Threshold the HLS image to get only white colors
                    mask = cv.inRange(hls, lower_white, upper_white)
                    # Bitwise-AND mask and original image
                    res = cv.bitwise_and(region, region, mask=mask)
                    if (visualization_phase):
                        cv.imshow("jump phase B/W image", res)
                        cv.waitKey(3)

                    jump, max_val_jump, _ = phase_find_best_match(region, [jump_check_icon], mask_icon)

                    if visualization_general_info:
                        print("jump check max_val: {:f} from file: {:s} round {:d}".format(max_val_jump, filename,
                                                                                           roundId))
                    if max_val_jump >= .58:
                        # if not in game recognise new round
                        if waiting4newround:
                            # outfile.write("\n\nNEW ROUND\t{:s}\n\n".format(frameTimeStamp))
                            if visualization_general_info:
                                print("|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NEW ROUND~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|")
                            last_known_valid_player_count = [100] * 3
                            last_recognised_player_count = [100] * 3
                            last_known_valid_kill_count = [0] * 3
                            last_recognised_kill_count = [0] * 3
                            avgLastPosition_icon = (lastPosition_icon[0], lastPosition_icon[1], 0)
                            lastPositions = []
                            waiting4newround = False
                            starttime = frameTimeStamp
                            startframe = frameId
                            roundId += 1
                            rnd_id = (filename.split("/")[-1]).split(".")[-2] + "_" + str(roundId)
                            if not os.path.exists(output_round + rnd_id + ".csv"):
                                outfile_round = open(output_round + rnd_id + ".csv", "a")
                                outfile_round.write("phase,x_pos,y_pos,distance,video_time,place,kills\n")
                                outfile_round.close()
                        # if in a game but player numbers are suddenly high again abort last round and start new round
                        elif frameId > startframe + frameRate * 60:
                            write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, -1,
                                      last_known_valid_player_count[-1], last_known_valid_kill_count[-1])
                            if visualization_general_info:
                                print(
                                    "|######################################ABORTED_GAME####################################|")
                                print("|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NEW ROUND~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|")
                            last_known_valid_player_count = [100] * 3
                            last_recognised_player_count = [100] * 3
                            last_known_valid_kill_count = [0] * 3
                            last_recognised_kill_count = [0] * 3
                            avgLastPosition_icon = (lastPosition_icon[0], lastPosition_icon[1], 0)
                            lastPositions = []
                            waiting4newround = False
                            starttime = frameTimeStamp
                            startframe = frameId
                            roundId += 1
                            rnd_id = (filename.split("/")[-1]).split(".")[-2] + "_" + str(roundId)
                            if not os.path.exists(output_round + rnd_id + ".csv"):
                                outfile_round = open(output_round + rnd_id + ".csv", "a")
                                outfile_round.write("phase,x_pos,y_pos,distance,video_time,place,kills\n")
                                outfile_round.close()

                # if player in round get map and game Data form the frame
                if not waiting4newround:
                    ### analyse map position

                    frameMap, x1, y1 = submatrix_video(frameV, (1018, 40), (1018 + 185, 40 + 185))
                    f = cv.cvtColor(frameMap, cv.COLORSPACE_RGBA)
                    # f = cv.resize(f, None, None, 0.66, 0.66, cv.INTER_CUBIC)

                    frame_green = f[:, :, 1]
                    frame_blue = f[:, :, 0]
                    frame_red = f[:, :, 2]

                    frame = cv.cvtColor(f, cv.COLOR_RGBA2GRAY)
                    lightness = np.mean(frame)

                    mg = np.mean(frame_green)
                    mr = np.mean(frame_red)
                    mb = np.mean(frame_blue)
                    frame = clahe.apply(frame)
                    # width & hight of the cropped part
                    w, h = frame.shape[::-1]
                    # print("w: {:d} h: {:d}".format(w, h))

                    # mr TODO mr & omitFrame = True?
                    if mg < 60:
                        omitFrame = True
                        continue
                    else:
                        omitFrame = False

                    # cv.imshow("videoframe", frame)
                    # cv.waitKey(1)

                    ### Apply template Matching
                    # make matrix smaller ...
                    if len(lastPositions) > 5 and not omitFrame:
                        # if we already have a few last positions we can cut down to a smaller map as there is not teleport available:
                        # submatrix size
                        n = 300
                        p = lastPositions[-1]
                        img_small, x1, y1 = submatrix(img_map, p, n)
                        res = cv.matchTemplate(img_small, frame, cv.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        top_left = (max_loc[0] + x1, max_loc[1] + y1)

                    else:
                        res = cv.matchTemplate(img_map, frame, cv.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                        top_left = max_loc

                    # check if it is on the island or not ...
                    # if img_mask[int(top_left[1] + w / 2), int(top_left[0] + h / 2)] < 128:
                    #    omitFrame = True
                    # print("frame {:06d} has mask {:03d}".format(count, img_mask[int(top_left[0] + w / 2), int(top_left[1] + h / 2)]))

                    if max_val > 0.35 and not omitFrame:
                        lastPositions.append(top_left)
                    else:  # reset last positions if match is not great
                        lastPositions = []

                    vis = vis_color.copy()

                    if len(lastPositions) > 10:
                        lastPositions = lastPositions[1:]

                    if len(lastPositions) > 1:
                        distance = np.math.sqrt(
                            (lastPositions[-2][0] - top_left[0]) ** 2 + (lastPositions[-2][1] - top_left[1]) ** 2)
                    else:
                        distance = -1

                    ### here is the text to recognize:
                    if output_read_num_next_to_icon:
                        print("players:")
                    count_players, count_img = read_num_next_to_icon(frameV, video_height, person_icon,
                                                                     person_icon_mask,
                                                                     lastPosition_icon, tool, builder)
                    if output_read_num_next_to_icon:
                        print("kills:")
                    count_player_kills, kills_img = read_num_next_to_icon(frameV, video_height, kills_icon,
                                                                          person_icon_mask,
                                                                          lastPosition_icon,
                                                                          tool,
                                                                          builder)

                    if output_read_num_next_to_icon and visualization_read_num_next_to_icon:
                        cv.imshow("player_count_img", count_img)
                        cv.imshow("kills_count_img", kills_img)
                        # cv.imwrite("kills_imgs/kills_img_{:06.0f}.png".format(frameId), kills_img)
                        cv.waitKey(1)

                    if len(count_players) > 2 or not pattern_player_count.match(count_players):
                        # print("count_player 2: {:s}".format(count_players))
                        count_players = -666

                    if len(count_player_kills) > 2 or not pattern_player_kills.match(count_player_kills):
                        # print("count_player 2: {:s}".format(count_players))
                        count_player_kills = -666

                    last_recognised_player_count.append(int(count_players))
                    last_recognised_player_count = last_recognised_player_count[1:]
                    last_known_valid_player_count = count_pattern_and_validity_check(last_known_valid_player_count,
                                                                                     last_recognised_player_count,
                                                                                     playerCountVarianz)

                    last_recognised_kill_count.append(int(count_player_kills))
                    last_recognised_kill_count = last_recognised_kill_count[1:]

                    last_known_valid_kill_count = count_pattern_and_validity_check(last_known_valid_kill_count,
                                                                                   last_recognised_kill_count,
                                                                                   playerKillVarianz)
                    if visualization_general_info:
                        print(
                            "{:.0f} {:s} {:2d} {:d} {:f} map position: {:d}x{:d} {:d}".format(frameId, frameTimeStamp,
                                                                                              current_phase,
                                                                                              last_known_valid_player_count[
                                                                                                  -1],
                                                                                              max_val,
                                                                                              top_left[0] + x1,
                                                                                              top_left[1] + y1,
                                                                                              last_known_valid_kill_count[
                                                                                                  -1]))

                    ######## finde Phase pt2 end

                    write_round(outfile_round, current_phase, int(top_left[0] + w / 2),
                                int(top_left[1] + h / 2), distance, frameTimeStamp, last_known_valid_player_count[-1],
                                last_known_valid_kill_count[-1])

                    if visualization_general_info:
                        print("{:4.0f} frames processed in {:.2f} seconds, making {:.2f} fps".format(frameId,
                                                                                                     time.time() - start_time,
                                                                                                     frameId / (
                                                                                                             time.time() - start_time)))
                    ### Visualisation map tracing
                    if (visualization_map):
                        for pos_idx in range(0, len(lastPositions)):
                            pos = lastPositions[pos_idx]
                            center = (int(pos[0] + w / 2), int(pos[1] + h / 2))
                            cv.circle(vis, center, pos_idx, (0, 0, 255, 50), -1)
                        if len(lastPositions) > 0:
                            pos = lastPositions[-1]
                            center = (int(pos[0] + w / 2), int(pos[1] + h / 2))
                            if not omitFrame:
                                heatmap[center[1], center[0]] = heatmap[center[1], center[0]] + 1
                                cv.circle(vis, center, 16, (0, 255, 255, 50), 2)
                            else:
                                cv.circle(vis, center, 16, (255, 0, 255, 50), 3)
                        cv.namedWindow('heatmap-result', cv.WINDOW_NORMAL)
                        cv.resizeWindow('heatmap-result', 1000, 1000)
                        # cv.imshow("heatmap-result", cv.resize(heatmap*255/np.max(heatmap), (0, 0), fx=0.75, fy=0.75))
                        # cv.imshow("heatmap-result", heatmap*255/np.max(heatmap))
                        cv.namedWindow('result', cv.WINDOW_NORMAL)
                        cv.resizeWindow('result', 1000, 1000)
                        cv.imshow("result", vis)
                        # cv.imshow("result", cv.resize(vis, (0, 0), fx=0.75, fy=0.75))
                        cv.imshow("map_frame", frame)
                        cv.waitKey(1)
    cap.release()
    if not waiting4newround:
        write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, -1,
                  last_known_valid_player_count[-1], last_known_valid_kill_count[-1])
        if visualization_general_info:
            print(
                "|######################################ABORTED_GAME####################################|")
