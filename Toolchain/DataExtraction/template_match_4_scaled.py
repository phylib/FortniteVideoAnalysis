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
    res = cv.matchTemplate(
        frameV[lastPosition_icon[1]:lastPosition_icon[1] + icon.shape[0], lastPosition_icon[0]:lastPosition_icon[0]+icon.shape[0]],
        icon,
        cv.TM_CCORR_NORMED, mask=icon_mask)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    text_img = frameV[max_loc[1] + lastPosition_icon[1]:max_loc[1] + lastPosition_icon[1] + icon.shape[0],
               max_loc[0] + lastPosition_icon[0] + icon.shape[0]:max_loc[0] + lastPosition_icon[0] + round(icon.shape[0]*2.5)]


    # show cropped image & place found icon position on the video frame
    if visualization_read_num_next_to_icon:
        cv.imshow("text search region", region)
        cv.imshow("text_img", text_img)
        vis = frameV.copy()
        cv.rectangle(vis, (lastPosition_icon[0] + max_loc[0], lastPosition_icon[1] + max_loc[1]),
                     (lastPosition_icon[0] + max_loc[0] + icon.shape[1],
                      lastPosition_icon[1] + max_loc[1] + icon.shape[0]),
                     255, 1)
        cv.imshow("funktion_icon", vis)
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


class MapLocation:

    def __init__(self, *args):
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
        self.scale = None
        if len(args) == 1:
            self.from_string(args[0])
        elif len(args) == 3:
            self.from_tuples(args[0], args[1], args[2])
        elif len(args) == 0:
            pass
        else:
            raise TypeError('huh ?.?')

    def from_tuples(self, top_left: (int, int), bottom_right: (int, int), scale: float):
        self.top = top_left[1]
        self.left = top_left[0]
        self.bottom = bottom_right[1]
        self.right = bottom_right[0]
        self.scale = scale
        return self

    def to_tuples(self):
        return (self.left, self.top), (self.right, self.bottom), self.scale

    def from_string(self, paras: str):
        paras = paras.replace("(","").replace(")","")
        sliced_paras = paras.split(",")
        if len(sliced_paras) == 5:
            self.left = int(sliced_paras[0])
            self.top = int(sliced_paras[1])
            self.right = int(sliced_paras[2])
            self.bottom = int(sliced_paras[3])
            self.scale = float(sliced_paras[4])
        return self

    def to_string(self):
        return "({:d},{:d}),({:d},{:d}),{:f}".format(self.left, self.top, self.right, self.bottom, self.scale)

    def compare_location(self, x: 'MapLocation'):
        return self.top == x.top and \
               self.left == x.left and \
               self.bottom == x.bottom and \
               self.right == x.right


def locate_map_scale(image):

    # shape[0] = height, shape[1] = width
    offsetX = int(image.shape[1] / 4) * 3
    offsetY = int(image.shape[0] / 2.5)
    image_crop, _, _ = submatrix_video(image, p1=(offsetX, 0), p2=(int(image.shape[1]), offsetY))

    # HLS
    lower_white = np.array([0, 190, 0]) #190 /200
    upper_white = np.array([255, 255, 255])

    hls = cv.cvtColor(image_crop, cv.COLOR_BGR2HLS)

    image_crop_L = cv.extractChannel(hls, 1)

    # Threshold the HLS image to get only white colors
    image_crop = cv.inRange(hls, lower_white, upper_white)

    if image.shape[0] == 720:
        count_low = 180 # 140
        count_high = 210 #
    else:
        count_low = 240 #280
        count_high = 305 #310
    res = []
    line_start = None
    # line_count = white pixel count of the line
    line_count = 0
    # line_length_count = line length including non white pixel gaps in the line
    line_length_count = 0
    # white_count = White pixel count of the cropped image
    white_count = 0
    # go through columns of the image
    for x in range(0, image_crop.shape[1]):
        # print("x: {:5d}/{:0.0f} \t count {:f} \t {:f} image_crop.shape[1]/8".format(x, (image_crop.shape[1]/2), count, (image_crop.shape[0] * image_crop.shape[1]) / 8))

        # only check left half of the image and
        # abort search if the image contains too much white (1/8 of the image in the left half)
        if x < (image_crop.shape[1]/2) and \
                white_count > ((image_crop.shape[0]*image_crop.shape[1])/8):
            if visualization_scalefinding:
                print("Scale None because white_count {:f} > {:f} image_crop.shape[1]/8".format(white_count, (image_crop.shape[0] * image_crop.shape[1]) / 8))
            return None
        # go through rows of the column
        for y in range(image_crop.shape[0]):
            # check if pixel is white
            if image_crop[y][x] == 255:
                white_count += 1
                line_count += 1
                line_length_count += 1
                if line_length_count == 1:
                    line_start = (x, y)
            else:
                # if line length is longer than 30 assume the line is not only noise or an icon
                if line_length_count < 30:
                    # if line_length_count >10:
                    #    print("line_length_count: {:d}".format(line_length_count))
                    line_start = None
                    line_length_count = 0
            # if line_length_count > 10:
                # print("line_length_count: {:d}".format(line_length_count))
        # check that the whole line is not white
        if count_low <= line_count <= count_high:
            if line_start is not None:
                if line_start[0] is not None and image_crop.shape[1] - line_count > line_start[0]:  # and line_start[1] > 10:
                    res.append((line_start, line_count, line_length_count))
        line_count = 0
        line_start = None
        line_length_count = 0

    if image.shape[0] == 1080:
        padding = 5
    else:
        padding = 4

    if visualization_scalefinding:
        print("white_count: {:d}".format(white_count))
        img_display = image.copy()
        # print("maxVal: {:f}\t frame h: {:f}\tscale: {:1.4f}\tratio: {:f}".format(maxVal, frame_h, scale, r))
        # print(res)
        # cv.resizeWindow('Image', 1000, 1000)
        cv.imshow("Scale + location result", img_display)
        # cv.imwrite(imagePath[0:-4]+"_found.png", image)
        cv.imshow('cropped b/w', image_crop)
        cv.imshow('cropped grey', image_crop_L)
        cv.waitKey(1)

    if len(res) > 0:
        temp = res[0]
        for x in range(1, len(res)):
            if res[x][1] > temp[1]:
                temp = res[x]

        top_left = (temp[0][0] + offsetX + padding, temp[0][1])
        bottom_right = (temp[0][0] + offsetX + temp[1] + padding, temp[0][1] + temp[1])
        scale = 286 / temp[1]

        if visualization_scalefinding:
            print("map length:", temp[1])
            print("scale:", scale)

        result = MapLocation(top_left, bottom_right, scale)
    else:
        result = None

    return result


# # File 1: all videos // won: -1 = abborted round; 0 = Player died; 1 = Player won
# def write_all(outfile, video, roundId, starttime, endtime, won, place, kills):
#     outfile = open(outfile.name, "a")
#     outfile.write(
#         "{:s},{:d},{:s},{:s},{:d},{:d},{:d},\n".format(video, roundId, starttime, endtime, won, place, kills))
#     outfile.close()
#     return None


# File 1: all videos // won: -1 = abborted round; 0 = Player died; 1 = Player won
def write_all(outfile, video, roundId, starttime, endtime, won, place, kills, final_map_location):
    outfile = open(outfile.name, "a")
    outfile.write(
        "{:s},{:d},{:s},{:s},{:d},{:d},{:d},({:d}x{:d})({:d}x{:d});{:6.3f},\n".format(video, roundId, starttime,
                                                                                      endtime, won, place, kills,
                                                                                      final_map_location.left,
                                                                                      final_map_location.top,
                                                                                      final_map_location.right,
                                                                                      final_map_location.bottom,
                                                                                      final_map_location.scale))
    outfile.close()
    return None


# File 2: per Video_RoundId:
def write_round(outfile, phase, x_pos, y_pos, distance, video_time, place, kills):
    outfile = open(outfile.name, "a")
    outfile.write(
        "{:1d},{:4d},{:4d},{:6.3f},{:s},{:d},{:d}\n".format(phase, x_pos, y_pos, distance, video_time, place, kills))
    outfile.close()
    return None


# File 2: per Video_RoundId:
def write_round_wo_placment(outfile, phase, x_pos, y_pos, distance, video_time, gliding, matching_val_map, matching_val_icon, omitFrame):
    #print("write_round_wo_placment entered")
    outfile = open(outfile.name, "a")
    outfile.write(
        "{:1d},{:4d},{:4d},{:6.3f},{:s},{:1d},{:6.3f},{:6.3f},{:1d}\n".format(phase, x_pos, y_pos, distance, video_time, gliding, matching_val_map, matching_val_icon, omitFrame))
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
visualization_scalefinding = False

### variable configure here...
# skip ahead some frames
jump_ahead_min = 3
jump_ahead_sec = 50


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to parse fortnite Videos.')
    parser.add_argument("-i", "--input", help="Video file to process")
    # parser.add_argument("-m", "--metaoutput", help="Output file for round meta information")
    # parser.add_argument("-r", "--roundoutput", help="Output file for detailed information off a round")
    parser.add_argument("-v", "--verbose", help="how many output to print, default, less", type=str2bool, default=False)
    parser.add_argument("-o", "--outpath", help="Path for output")
    parser.add_argument("-t", "--testrun", help="debug test run mode for single videos", type=str2bool, default=False)

    args = parser.parse_args()
    filename = args.input
    # output_all = args.metaoutput
    # output_round = args.roundoutput
    verbose = args.verbose
    output_path = args.outpath
    testrun = args.testrun

    # print(verbose)
    if not verbose:
        visualization_map = False
        visualization_phase = False
        visualization_deathORwin = False
        visualization_read_num_next_to_icon = False
        output_read_num_next_to_icon = False
        visualization_general_info = False
        visualization_scalefinding = False

    # print((filename, verbose, output_path, testrun))
    print("Starting video: {:s} ".format(filename))

    if testrun:
        output_timestamp = time.strftime("_%d-%m-%y_%H.%M.%S", time.gmtime())
    else:
        # output_timestamp = time.strftime("%d-%m-%y", time.gmtime())
        output_timestamp = ""

    dirName = output_path + '/fn-dataextraction'
    try:
        os.mkdir(dirName)
        if visualization_general_info:
            print("Directory ", dirName, " Created ")
    except FileExistsError:
        if visualization_general_info:
            print("Directory ", dirName, " already exists")

    dirName = dirName + '/fn-dataextraction' + output_timestamp
    try:
        os.mkdir(dirName)
        if visualization_general_info:
            print("Directory ", dirName, " Created ")
    except FileExistsError:
        if visualization_general_info:
            print("Directory ", dirName, " already exists")

    output_all = dirName + '/match-overview.csv'
    output_round = dirName + '/round-'

    # setting up map for position tracing
    img_map = cv.imread('Chapter_2_Season_1_Minimap_2000x2000.png', cv.IMREAD_GRAYSCALE)
    img_map_mask = cv.imread('map_mask_ch2_2000x2000.png', cv.IMREAD_GRAYSCALE)
    vis_color = cv.imread('Chapter_2_Season_1_Minimap_2000x2000.png', cv.IMREAD_COLOR)
    # vis_color = cv.resize(vis_color, (1550, 1550), interpolation=cv.INTER_CUBIC) # Scale ok
    # vis_color = cv.resize(vis_color, None, None, 1.39, 1.39, cv.INTER_CUBIC) # Scale ok
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_map = clahe.apply(img_map)

    heatmap = np.zeros(img_map_mask.shape)

    lastPositions = []

    start_time = time.time()

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
    #person_icon_mask = cv.imread("person_icon_mask.png")

    #person_icon = cv.imread("person_icon.png")

    #kills_icon = cv.imread("kills_icon_ch2.png")

    phase_icons = [cv.imread("jump_icon.png", cv.IMREAD_COLOR),
                      cv.imread("time_icon_.png", cv.IMREAD_COLOR),
                      cv.imread("storm_icon.png", cv.IMREAD_COLOR)]

    # image for check if jump is really jump
    jump_check_icon = cv.imread("jump_check_icon.png", cv.IMREAD_COLOR)

    # resize images for base scaling
    icon_size = 28
    for pid in range(0, len(phase_icons)):
        phase_icons[pid] = cv.resize(phase_icons[pid], (icon_size, icon_size), interpolation=cv.INTER_CUBIC)
    #print(phase_icons[0].shape[0])
    jump_check_icon = cv.resize(jump_check_icon, (icon_size, icon_size), interpolation=cv.INTER_CUBIC)
    mask_icon = cv.resize(mask_icon, (icon_size, icon_size), interpolation=cv.INTER_CUBIC)
    #person_icon_mask = cv.resize(person_icon_mask, (icon_size, icon_size), interpolation=cv.INTER_CUBIC)
    #person_icon = cv.resize(person_icon, (icon_size, icon_size), interpolation=cv.INTER_CUBIC)
    #kills_icon = cv.resize(kills_icon, (icon_size, icon_size), interpolation=cv.INTER_CUBIC)

    win_img = cv.imread("win_1080.png")
    win_img = cv.resize(win_img, (675, 230), interpolation=cv.INTER_CUBIC)
    win_img_mask = cv.imread("win_1080_mask.png")
    win_img_mask = cv.resize(win_img_mask, (675, 230), interpolation=cv.INTER_CUBIC)
    death_img = cv.imread("youPlaced.png")
    death_mask = cv.imread("youPlaced_mask.png")

    # last positions of icon (x,y)
    lastPosition_icon = (0, 0)
    # avg last positions of icon (x, y, count)
    ###avgLastPosition_icon = (0, 0, 0)
    # samePositionCount = 0
    current_phase = -1

    #
    text_region_h = None

    # new round starting?
    waiting4newround = True
    roundId = 0

    if not os.path.exists(output_all):
        outfile_all = open(output_all, "a")
        outfile_all.write("video,roundId,starttime,endtime,won,place,kills,map_ui_position/scale\n")
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

    map_locations_list = []
    # scale to scale up Video to 1080p-icons recognition
    scale = None
    # frame_ height that the scale was found for (some VODs change frame size)
    frame_height = None
    final_map_location = None
    final_map_location_rescaled = False

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

            if final_map_location is None:
                # cut search region and find out phase
                map_location = locate_map_scale(frameV)

                # add found map region to map_locations
                if map_location is not None:
                    map_locations_list.append((map_location, frameTime))
                    if len(map_locations_list) >=60:
                        map_locations_list = map_locations_list[-30:]

                # if gap between recognised bars is longer than 5 sec. reset map_locations
                if len(map_locations_list) >= 2 and (map_locations_list[-1][1] - map_locations_list[-2][1]) > 5:
                    map_locations_list = map_locations_list[-1:]

                # print("map_locations:{:2d}".format(len(map_locations_list)))
                # temp = ''
                # for i in range(len(map_locations_list)):
                #     temp += map_locations_list[i][0].to_string()
                # print(temp)
                if len(map_locations_list) >= 5:
                    for region in map_locations_list:
                        voter = {}
                        for i in range(0, len(map_locations_list)):
                            if map_locations_list[i][0].to_string() not in voter:
                                voter[map_locations_list[i][0].to_string()] = 1
                            else:
                                voter[map_locations_list[i][0].to_string()] += 1

                    count = sorted(voter, key=voter.get)#[0]
                    # print("count len:{:2d}".format(len(count)))
                    # print(count)
                    # print("voter:")
                    # print(voter)
                    final_map_location = MapLocation().from_string(count[0])

                if final_map_location is not None:
                    frame_height = video_height
                    region_top_left, region_bottom_right, scale = final_map_location.to_tuples()

                    region, x1, y1 = submatrix_video(frameV, region_top_left, (region_bottom_right[0], region_bottom_right[1]))

                    if visualization_scalefinding:
                        cv.imshow('first map cutout', region)
                        cv.waitKey(1)
                        cv.destroyWindow('first map cutout')

                    map_length = final_map_location.right - final_map_location.left
                    icon_size = round(map_length*.1)

            # skip frame if scale is known but video size changed
            elif frame_height != video_height:
                # Abort round if currently in a one
                if not waiting4newround:
                    write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, -1,
                              last_known_valid_player_count[-1], last_known_valid_kill_count[-1],
                              final_map_location)
                    if visualization_general_info:
                        print(
                            "|######################################ABORTED_GAME_2####################################|")
                        print("|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NEW ROUND 1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|")

                    last_known_valid_player_count = [100] * 3
                    last_recognised_player_count = [100] * 3
                    last_known_valid_kill_count = [0] * 3
                    last_recognised_kill_count = [0] * 3
                    # avgLastPosition_icon = (lastPosition_icon[0], lastPosition_icon[1], 0)
                    lastPositions = []
                    waiting4newround = True
                    starttime = ""
                    startframe = frameId
                    roundId += 1
                    rnd_id = (filename.split("/")[-1]).split(".")[-2] + "_" + str(roundId)
                    if not os.path.exists(output_round + rnd_id + ".csv"):
                        outfile_round = open(output_round + rnd_id + ".csv", "a")
                        outfile_round.write("phase,x_pos,y_pos,distance,video_time,gliding,matching_val_map,matching_val_phase,omitFrame\n")
                        outfile_round.close()
            # Scale is known
            else:
                #print("scale frame h: {:f}, cur frame h: {:f}".format(frame_height, video_height))
                region_top_left, region_bottom_right, scale = final_map_location.to_tuples()

                # resize frame and set finalmaplocation
                if (not final_map_location_rescaled):
                    final_map_location.left = round(final_map_location.left * scale)
                    final_map_location.right = round(final_map_location.right * scale)
                    final_map_location.top = round(final_map_location.top * scale)
                    final_map_location.bottom = round(final_map_location.bottom * scale)
                    text_region_h = int((jump_check_icon.shape[0] + 10) * scale)
                    final_map_location_rescaled = True

                frameV = cv.resize(frameV, None, None, final_map_location.scale, final_map_location.scale, cv.INTER_AREA)

                # print(final_map_location.left, final_map_location.bottom),
                (final_map_location.right,
                 final_map_location.bottom + text_region_h)

                '''region, x1, y1 = submatrix_video(frameV, (final_map_location.left, final_map_location.bottom),
                                                 (final_map_location.right,
                                                  final_map_location.bottom + text_region_h))'''

                region, x1, y1 = submatrix_video(frameV, (final_map_location.left, final_map_location.bottom),
                                                 (final_map_location.right-int((final_map_location.right-final_map_location.left)/2),
                                                  final_map_location.bottom + text_region_h))

                #current_phase, max_val_phase, max_loc = phase_find_best_match(region, phase_icons, mask_icon)
                current_phase, max_val_phase, max_loc = phase_find_best_match(region, phase_icons, mask_icon)

                if visualization_general_info:
                    # if no matching phase was found set phase to -1
                    print("max_val_phase: {:f} currentphase found: {:d}".format(max_val_phase, current_phase))

                #if (current_phase == 0 and max_val_phase < .91) or max_val_phase < .966:  # .96 --> .954 --> .966
                if max_val_phase < .946:
                    if visualization_general_info:
                        print("current_phase set from {:f} to -1".format(max_val_phase))
                    current_phase = -1

                lastPosition_icon = (max_loc[0] + x1, max_loc[1] + y1)
                top_left = (max_loc[0], max_loc[1])

                if visualization_phase:
                    vis = frameV.copy()
                    cv.rectangle(vis, (top_left[0] + x1, top_left[1] + y1),
                                 (top_left[0] + x1 + mask_icon.shape[1],
                                  top_left[1] + y1 + mask_icon.shape[0]),
                                 255, 3)
                    cv.imshow("video", vis)
                    cv.imshow("region_Phase", region)
                    cv.waitKey(1)

                # if in a round and no phase was found: check if player died/won
                if not waiting4newround:
                    #if current_phase <= 0:
                    if current_phase <= 0:
                        '''if ((current_phase <= 0):
                                or ((lastPosition_icon[0] <= avgLastPosition_icon[0] - 2 or
                                     lastPosition_icon[0] >= avgLastPosition_icon[0] + 2)
                                    or (lastPosition_icon[1] >= avgLastPosition_icon[1] + 2 or
                                        lastPosition_icon[1] <= avgLastPosition_icon[1] - 2))):'''
                        # HLS
                        lower_white = np.array([0, 230, 0])
                        upper_white = np.array([255, 255, 255])

                        # Check if player died
                        region_, x1_, y1_ = submatrix_video(frameV, (600, 100), (600 + 700, 100 + 300))
                        hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                        # Threshold the HLS image to get only white colors
                        mask = cv.inRange(hls, lower_white, upper_white)

                        # Bitwise-AND mask and original image to keep slight color in whites of the image
                        res = cv.bitwise_and(region_, region_, mask=mask)

                        death, max_val_death, max_loc_ = phase_find_best_match(res, [death_img], death_mask)

                        if visualization_deathORwin:
                            # cv.imshow("deathframecrop", res)
                            print("Max_val_ death: {:f}".format(max_val_death))
                        # if max_val_death of player death high enough write that player died
                        if max_val_death >= .80:

                            write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, 0,
                                      last_known_valid_player_count[-1], last_known_valid_kill_count[-1], final_map_location)
                            waiting4newround = True
                            if visualization_deathORwin:
                                print(
                                    "|#############################################DEATH####################################|")

                        # check if player won
                        else:
                            region_, x1_, y1_ = submatrix_video(frameV, (450, 70), (450 + 1000, 80 + 400)) # 710, 250

                            # hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)
                            #
                            # # Threshold the HLS image to get only white colors
                            # mask = cv.inRange(hls, lower_white, upper_white)
                            # # Bitwise-AND mask and original image
                            # res = cv.bitwise_and(region_, region_, mask=mask)

                            win, max_val_win, max_loc_ = phase_find_best_match(region_, [win_img], win_img_mask)

                            if visualization_deathORwin:
                                # cv.imshow("winframecrop", res)
                                # cv.imwrite("WinCrop_{:06.0f}.png".format(frameId), res)
                                print("Max_val_win: {:f}".format(max_val_win))
                            # if max_val_win of player won high enough write that player won
                            if max_val_win >= .9:

                                # if only 2 players are left and player wins -> one more kill to the count
                                corrected_win_kills = last_known_valid_kill_count[-1]
                                corrected_player_count = last_known_valid_player_count[-1]

                                if last_known_valid_player_count[-1] == 2:
                                    corrected_win_kills = corrected_win_kills + 1

                                # if win not imideatly recognissed & updated count recognised -> +1 to playercount to correct output
                                elif last_known_valid_player_count == 1:
                                    corrected_player_count += 1

                                write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, 1,
                                          1, -1, final_map_location)
                                waiting4newround = True
                                if visualization_deathORwin:
                                    print(
                                        "|#############################################WIN####################################|")

                '''# Update avg last pos after useing it in check
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
                                                                                            lastPosition_icon[1]))'''
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
                        cv.waitKey(1)

                    jump, max_val_jump, _ = phase_find_best_match(region, [jump_check_icon], mask_icon)

                    if visualization_general_info:
                        print("jump check max_val_jump: {:f} from file: {:s} round {:d}".format(max_val_jump, filename,
                                                                                                roundId))
                    if max_val_jump >= .58:
                        # if not in game recognise new round
                        if waiting4newround:
                            # outfile.write("\n\nNEW ROUND\t{:s}\n\n".format(frameTimeStamp))

                            last_known_valid_player_count = [100] * 3
                            last_recognised_player_count = [100] * 3
                            last_known_valid_kill_count = [0] * 3
                            last_recognised_kill_count = [0] * 3
                            # avgLastPosition_icon = (lastPosition_icon[0], lastPosition_icon[1], 0)
                            lastPositions = []
                            waiting4newround = False
                            starttime = frameTimeStamp
                            startframe = frameId
                            roundId += 1
                            rnd_id = (filename.split("/")[-1]).split(".")[-2] + "_" + str(roundId)
                            if not os.path.exists(output_round + rnd_id + ".csv"):
                                outfile_round = open(output_round + rnd_id + ".csv", "a")
                                outfile_round.write("phase,x_pos,y_pos,distance,video_time,gliding,matching_val_map,matching_val_phase,omitFrame\n")
                                outfile_round.close()

                            if visualization_general_info:
                                print("|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NEW ROUND 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|")
                        # It is 60 seconds after the initial jump detection

                        # TODO remove this abort round?
                        elif frameId > startframe + frameRate * 60:
                            write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, -1,
                                      last_known_valid_player_count[-1], last_known_valid_kill_count[-1],
                                      final_map_location)

                            last_known_valid_player_count = [100] * 3
                            last_recognised_player_count = [100] * 3
                            last_known_valid_kill_count = [0] * 3
                            last_recognised_kill_count = [0] * 3
                            # avgLastPosition_icon = (lastPosition_icon[0], lastPosition_icon[1], 0)
                            lastPositions = []
                            waiting4newround = False
                            starttime = frameTimeStamp
                            startframe = frameId
                            roundId += 1
                            rnd_id = (filename.split("/")[-1]).split(".")[-2] + "_" + str(roundId)
                            if not os.path.exists(output_round + rnd_id + ".csv"):
                                outfile_round = open(output_round + rnd_id + ".csv", "a")
                                outfile_round.write("phase,x_pos,y_pos,distance,video_time,gliding,matching_val_map,matching_val_phase,omitFrame\n")
                                outfile_round.close()

                            if visualization_general_info:
                                print(
                                    "|######################################ABORTED_GAME####################################|")
                                print("|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NEW ROUND 3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|")

                # if player in round get map and game Data form the frame
                if not waiting4newround:
                    gliding = -1
                    # Check if
                    if locate_map_scale(frameV) is None:
                        gliding = 0
                    else:
                        gliding = 1

                    ### analyse map position
                    frameMap, x1, y1 = submatrix_video(frameV, region_top_left, (region_bottom_right[0], region_bottom_right[1]))
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

                    if mg < 60:
                        omitFrame = True
                        continue
                    else:
                        omitFrame = False

                    # cv.imshow("videoframe", frame)
                    # cv.waitKey(1)

                    ### Apply template Matching
                    # make matrix smaller ...
                    if len(lastPositions) > 5: # and not omitFrame:
                        # if we already have a few last positions we can cut down to a smaller map as there is not teleport available:
                        # submatrix size
                        n = 300
                        p = lastPositions[-1]
                        img_small, x1, y1 = submatrix(img_map, p, n)
                        res = cv.matchTemplate(img_small, frame, cv.TM_CCOEFF_NORMED)
                        min_val, max_val_map, min_loc, max_loc = cv.minMaxLoc(res)
                        top_left = (max_loc[0] + x1, max_loc[1] + y1)

                    else:
                        res = cv.matchTemplate(img_map, frame, cv.TM_CCOEFF_NORMED)
                        min_val, max_val_map, min_loc, max_loc = cv.minMaxLoc(res)
                        top_left = max_loc

                    # check if it is on the island or not ...
                    if img_map_mask[int(top_left[1] + w / 2), int(top_left[0] + h / 2)] < 128:
                        omitFrame = True
                        # print("frame {:06d} has mask {:03d}".format(count, img_map_mask[int(top_left[0] + w / 2), int(top_left[1] + h / 2)]))

                    if max_val_map > 0.35 and not omitFrame:
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

                    # ### here is the text to recognize:
                    # if output_read_num_next_to_icon:
                    #     print("players:")
                    # count_players, count_img = read_num_next_to_icon(frameV, video_height, person_icon,
                    #                                                  person_icon_mask,
                    #                                                  lastPosition_icon, tool, builder)
                    # if output_read_num_next_to_icon:
                    #     print("kills:")
                    # count_player_kills, kills_img = read_num_next_to_icon(frameV, video_height, kills_icon,
                    #                                                       person_icon_mask,
                    #                                                       lastPosition_icon,
                    #                                                       tool,
                    #                                                       builder)
                    #
                    # if output_read_num_next_to_icon and visualization_read_num_next_to_icon:
                    #     cv.imshow("player_count_img", count_img)
                    #     cv.imshow("kills_count_img", kills_img)
                    #     # cv.imwrite("kills_imgs/kills_img_{:06.0f}.png".format(frameId), kills_img)
                    #     cv.waitKey(1)


                    # if len(count_players) > 2 or not pattern_player_count.match(count_players) and not count_players==100:
                    #     # print("count_player 2: {:s}".format(count_players))
                    #     count_players = -666
                    #
                    # if len(count_player_kills) > 2 or not pattern_player_kills.match(count_player_kills):
                    #     # print("count_player 2: {:s}".format(count_players))
                    #     count_player_kills = -666

                    # last_recognised_player_count.append(int(count_players))
                    # last_recognised_player_count = last_recognised_player_count[1:]
                    # last_known_valid_player_count = count_pattern_and_validity_check(last_known_valid_player_count,
                    #                                                                  last_recognised_player_count,
                    #                                                                  playerCountVarianz)
                    #
                    # last_recognised_kill_count.append(int(count_player_kills))
                    # last_recognised_kill_count = last_recognised_kill_count[1:]

                    # last_known_valid_kill_count = count_pattern_and_validity_check(last_known_valid_kill_count,
                    #                                                                last_recognised_kill_count,
                    #                                                                playerKillVarianz)
                    if visualization_general_info:
                        # print(
                        #     "{:.0f} {:s} {:2d} {:d} {:f} map position: {:d}x{:d} {:d}".format(frameId, frameTimeStamp,
                        #                                                                       current_phase,
                        #                                                                       last_known_valid_player_count[
                        #                                                                           -1],
                        #                                                                       max_val,
                        #                                                                       top_left[0] + x1,
                        #                                                                       top_left[1] + y1,
                        #                                                                       last_known_valid_kill_count[
                        #                                                                           -1]))
                        print(
                            "{:.0f} {:s} {:2d} Map_max_val: {:f} map position: {:d}x{:d}".format(frameId, frameTimeStamp,
                                                                                              current_phase,
                                                                                              max_val_map,
                                                                                              top_left[0] + x1,
                                                                                              top_left[1] + y1))

                    ######## finde Phase pt2 end

                    # write_round(outfile_round, current_phase, int(top_left[0] + w / 2),
                    #             int(top_left[1] + h / 2), distance, frameTimeStamp,
                    #             last_known_valid_player_count[-1],
                    #             last_known_valid_kill_count[-1])

                    if outfile_round is None:
                        # outfile_round already existed as round started and will not be altered
                        print("video: {:s}\troundId: {:d}".format(filename, roundId))

                    else:
                        write_round_wo_placment(outfile_round, current_phase, int(top_left[0] + w / 2),
                                                int(top_left[1] + h / 2), distance, frameTimeStamp, gliding,
                                                max_val_map, max_val_phase, omitFrame)

                    if visualization_general_info:
                        print("{:4.0f} frames processed in {:.2f} seconds, making {:.2f} fps".format(frameId,
                                                                                                     time.time() - start_time,
                                                                                                     frameId / (
                                                                                                                 time.time() - start_time)))
                    ### Visualisation map tracing
                    if (visualization_map):
                        for pos_idx in range(0, len(lastPositions)):
                            pos = lastPositions[pos_idx]
                            center = (pos[0] + int(w / 2), pos[1] + int(h / 2))
                            cv.circle(vis, center, pos_idx, (0, 0, 255, 50), -1)
                        if len(lastPositions) > 0:
                            pos = lastPositions[-1]
                            center = (pos[0] + int(w / 2), pos[1] + int(h / 2))
                            if not omitFrame:
                                heatmap[center[1], center[0]] = heatmap[center[1], center[0]] + 1
                                cv.circle(vis, center, 16, (0, 255, 255, 50), 2)
                            else:
                                cv.circle(vis, center, 16, (255, 0, 255, 50), 3)
                        #cv.namedWindow('heatmap-result', cv.WINDOW_NORMAL)
                        #cv.resizeWindow('heatmap-result', 1000, 1000)
                        # cv.imshow("heatmap-result", cv.resize(heatmap*255/np.max(heatmap), (0, 0), fx=0.75, fy=0.75))
                        # cv.imshow("heatmap-result", heatmap*255/np.max(heatmap))
                        cv.namedWindow('result', cv.WINDOW_NORMAL)
                        cv.resizeWindow('result', 1000, 1000)
                        cv.imshow("result", vis)
                        print("vissize:",vis.shape[0])
                        # cv.imshow("result", cv.resize(vis, (0, 0), fx=0.75, fy=0.75))
                        cv.imshow("map_frame", frame)
                        cv.waitKey(1)
    cap.release()
    if final_map_location is None:
        print("End of video: {:s} \t No final_map_location found.".format(filename))
    else:
        print("End of video: {:s} ".format(filename))

    if not waiting4newround:
        write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, -1,
                  -1, -1, final_map_location)
        if visualization_general_info:
            print(
                "|######################################ABORTED_GAME####################################|")

