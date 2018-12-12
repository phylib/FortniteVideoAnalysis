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
    max_idx = np.argmax(max_values)
    return max_idx, max_values[max_idx], max_locations[max_idx]


def read_num_next_to_icon(frameV, frameV_size, icon, icon_mask, lastPosition_icon, tool, builder):
    # crop region for searching according to video size
    if frameV_size == 720:
        res = cv.matchTemplate(
            frameV[lastPosition_icon[1]:lastPosition_icon[1] + 25, lastPosition_icon[0]:1215],
            icon,
            cv.TM_CCORR_NORMED, mask=icon_mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        text_img = frameV[max_loc[1] + lastPosition_icon[1]:max_loc[1] + lastPosition_icon[1] + 20,
                   max_loc[0] + lastPosition_icon[0] + 24:max_loc[0] + lastPosition_icon[0] + 45]
    else:
        res = cv.matchTemplate(
            frameV[lastPosition_icon[1]:lastPosition_icon[1] + 45, lastPosition_icon[0]:1810],
            icon,
            cv.TM_CCORR_NORMED, mask=icon_mask)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        text_img = frameV[max_loc[1] + lastPosition_icon[1]:max_loc[1] + lastPosition_icon[1] + 33,
                   max_loc[0] + lastPosition_icon[0] + 36:max_loc[0] + lastPosition_icon[
                       0] + 70]
    # show cropped image & place found icon position on the video frame
    if visualization_read_num_next_to_icon:
        cv.imshow("text search region", region)
        cv.imshow("text_img", text_img)
        cv.rectangle(frameV, (lastPosition_icon[0] + max_loc[0], lastPosition_icon[1] + max_loc[1]),
                     (lastPosition_icon[0] + max_loc[0] + icon.shape[1],
                      lastPosition_icon[1] + max_loc[1] + icon.shape[0]),
                     255, 3)
        # cv.imshow("funktion_icon", frameV)
        cv.waitKey(1)
    # read the text next to the icon
    text_img = cv.cvtColor(text_img, cv.COLOR_BGR2GRAY)
    text_img = cv.bilateralFilter(text_img, 5, 100, 100)
    text_img = cv.equalizeHist(text_img)
    res_img = cv.threshold(text_img, 200, 255, cv.THRESH_BINARY_INV)[1]
    pil_im = Image.fromarray(res_img)

    # Setting in tesseract for digits recognition(-> changing the page segmentation mode)
    builder.tesseract_flags = ['-psm', '3']  # If tool = libtesseract
    number = tool.image_to_string(
        pil_im,
        lang=lang,
        builder=builder
    ).replace('.', ' ').replace('-', ' ').strip()
    # if no digit recognised or wrong one try setting reading for 1 digit
    if number == "" or len(number) > 2:
        # Setting in tesseract for single digit recognition-> changing the page segmentation mode
        builder.tesseract_flags = ['-psm', '10']  # If tool = libtesseract

        res_img = cv.threshold(text_img, 225, 255, cv.THRESH_BINARY_INV)[1]
        # cv.threshold(text_img, 200, 255, cv.THRESH_BINARY_INV)[1]
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
        elif newNr == last_recognised_count[-2] == last_recognised_count[-3]:# \
                #and int(last_known_valid_numbers[-1]) - varianz_down_big <= int(newNr) \
                #<= int(last_known_valid_numbers[-1]) + varianz_up_big:
            last_known_valid_numbers.append(newNr)
            last_known_valid_numbers = last_known_valid_numbers[1:]

    else:  # TODO if last numbers are not the same add some flag?
        # check if last 2 recog. nr are constant and stick to that
        if last_recognised_count[-2] == last_recognised_count[-3] != -666:
            last_known_valid_numbers.append(last_recognised_count[-2])
            last_known_valid_numbers = last_known_valid_numbers[1:]

    if output_read_num_next_to_icon:
        print('after:', last_known_valid_numbers, last_recognised_count)

    return last_known_valid_numbers


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

visualization_map = False
visualization_phase = False
visualization_deathORwin = True
visualization_read_num_next_to_icon = False
output_read_num_next_to_icon = False
visualization_general_info = True

### variable configure here...
# skip ahead some frames
jump_ahead_min = 0
jump_ahead_sec = 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to parse fortnite Videos.')
    parser.add_argument("-i", "--input", help="Video file to process")
    parser.add_argument("-m", "--metaoutput", help="Output file for round meta information")
    parser.add_argument("-r", "--roundoutput", help="Output file for detailed information off a round")
    parser.add_argument("-v", "--verbose", help="how many output to print, default, less", type=bool, default=False)

    args = parser.parse_args()
    filename = args.input
    output_all = args.metaoutput
    output_round = args.roundoutput
    verbose = args.verbose

    if not verbose:
        visualization_map = False
        visualization_phase = False
        visualization_deathORwin = False
        visualization_read_num_next_to_icon = False
        output_read_num_next_to_icon = False
        visualization_general_info = False

    # setting up map for position tracing
    img = cv.imread('map.png', cv.IMREAD_GRAYSCALE)
    img_mask = cv.imread('map_mask.png', cv.IMREAD_GRAYSCALE)
    vis_color = cv.imread('map.png', cv.IMREAD_COLOR)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    heatmap = np.zeros(img_mask.shape)

    lastPositions = []

    start_time = time.time()

    # check if video is form UseCaseStudy -> different cropping settings
    if filename.__contains__("Fn"):
        FnUseCaseStudy = True
    else:
        FnUseCaseStudy = False

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

    # Phases index
    # 0 - jump
    # 1 - time till contract
    # 2 - contracting
    # TODO add in bus phase again?

    # images for phase recognition
    mask_icon720 = cv.imread("mask_icon.png")
    mask_icon1080 = cv.resize(mask_icon720, None, None, 1.5, 1.5, cv.INTER_CUBIC)
    person_icon_mask720 = cv.imread("person_icon_mask.png")
    person_icon_mask1080 = cv.resize(person_icon_mask720, None, None, 1.5, 1.5, cv.INTER_CUBIC)

    person_icon720 = cv.imread("person_icon.png")
    person_icon1080 = cv.resize(person_icon720, None, None, 1.5, 1.5, cv.INTER_CUBIC)

    kills_icon720 = cv.imread("kills_icon.png")
    kills_icon1080 = cv.resize(kills_icon720, None, None, 1.5, 1.5, cv.INTER_CUBIC)

    phase_icons720 = [cv.imread("jump_icon.png", cv.IMREAD_COLOR),
                      cv.imread("time_icon_.png", cv.IMREAD_COLOR),
                      cv.imread("storm_icon.png", cv.IMREAD_COLOR)]

    # image for check if jump is really jump
    jump_check_icon720 = cv.imread("jump_check_icon.png", cv.IMREAD_COLOR)
    jump_check_icon1080 = cv.resize(jump_check_icon720, None, None, 1.5, 1.5, cv.INTER_CUBIC)

    phase_icons1080 = []
    for pid in range(0, len(phase_icons720)):
        phase_icons1080.append(cv.resize(phase_icons720[pid], None, None, 1.5, 1.5, cv.INTER_CUBIC))
        # cv.imshow("icon{:d}".format(pid), phase_icons1080[pid])
        # cv.imshow("iconorig{:d}".format(pid), phase_icons720[pid])

    # last positions of icon (x,y)
    lastPosition_icon = (0, 0)
    # avg last positions of icon (x, y, count)
    avgLastPosition_icon = (0, 0, 0)
    # samePositionCount = 0
    current_phase = -1

    win_mask_720 = cv.imread("win_mask_720_.png")
    win_mask_1080 = cv.resize(win_mask_720, None, None, 1.5, 1.5, cv.INTER_CUBIC)
    death_mask_720 = cv.imread("death_mask_720_all.png")
    death_mask_1080 = cv.resize(death_mask_720, None, None, 1.5, 1.5, cv.INTER_CUBIC)

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
    playerCountVarianz = (4, 3, 35, 35)  # TODO check if big var up 35 okay

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

    # go through frame by frame
    while cap.isOpened() and ret:
        # get next frame
        ret, frameV = cap.read()
        if not ret:
            break

        ### Set variables according to video size (720p or 1080p)
        frameTime = math.floor(cap.get(cv.CAP_PROP_POS_MSEC) / 1000)
        video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float

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
                    "\nframeid: %i out of %i\tTimestamp: %s" % (
                    frameId, cap.get(cv.CAP_PROP_FRAME_COUNT), frameTimeStamp))
            ######## finde Phase pt2

            # cut search region and find out phase
            if video_height == 720 and FnUseCaseStudy:
                region, x1, y1 = submatrix_video(frameV, (1010, 220), (1100, 250))  # 1100        1170
                current_phase, max_val, max_loc = phase_find_best_match(region, phase_icons720, mask_icon720)

            elif video_height == 720:
                region, x1, y1 = submatrix_video(frameV, (1015, 200), (1080, 260))  # 1100        1200
                current_phase, max_val, max_loc = phase_find_best_match(region, phase_icons720, mask_icon720)

            else:
                region, x1, y1 = submatrix_video(frameV, (1525, 335), (1650, 385))  # 1650        1800
                current_phase, max_val, max_loc = phase_find_best_match(region, phase_icons1080, mask_icon1080)

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
                if video_height == 720:
                    cv.rectangle(frameV, (top_left[0] + x1, top_left[1] + y1),
                                 (top_left[0] + x1 + mask_icon720.shape[1],
                                  top_left[1] + y1 + mask_icon720.shape[0]),
                                 255, 3)
                else:
                    cv.rectangle(frameV, (top_left[0] + x1, top_left[1] + y1),
                                 (top_left[0] + x1 + mask_icon1080.shape[1],
                                  top_left[1] + y1 + mask_icon1080.shape[0]),
                                 255, 3)
                cv.imshow("video", frameV)
                cv.imshow("region_Phase", region)
                cv.waitKey(3)

            # if in a round and no phase was found or icon position changed: check if player died/won
            if not waiting4newround:
                if ((current_phase <= 0)
                        or ((lastPosition_icon[0] <= avgLastPosition_icon[0] - 2 or
                                lastPosition_icon[0] >= avgLastPosition_icon[0] + 2)
                            or (lastPosition_icon[1] >= avgLastPosition_icon[1] + 2 or
                                lastPosition_icon[1] <= avgLastPosition_icon[1]-2))):
                    # HLS
                    lower_white = np.array([0, 230, 0])
                    upper_white = np.array([255, 255, 255])

                    # Check if player died
                    if video_height == 720:
                        region_, x1_, y1_ = submatrix_video(frameV, (450, 95), (450 + 360, 95 + 60))
                        hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                        # Threshold the HLS image to get only white colors
                        mask = cv.inRange(hls, lower_white, upper_white)
                        # Bitwise-AND mask and original image
                        res = cv.bitwise_and(region_, region_, mask=mask)

                        death, max_val_, max_loc_ = phase_find_best_match(res, [death_mask_720], None)

                    else:
                        region_, x1_, y1_ = submatrix_video(frameV, (700, 150), (745 + 515, 150 + 75))
                        hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                        # Threshold the HLS image to get only white colors
                        mask = cv.inRange(hls, lower_white, upper_white)
                        # Bitwise-AND mask and original image
                        res = cv.bitwise_and(region_, region_, mask=mask)

                        death, max_val_, max_loc_ = phase_find_best_match(res, [death_mask_1080], None)
                    if visualization_deathORwin:
                        #cv.imshow("deathframecrop", res)
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
                        if video_height == 720:
                            region_, x1_, y1_ = submatrix_video(frameV, (325, 120), (325 + 535, 120 + 110))
                            hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                            # Threshold the HLS image to get only white colors
                            mask = cv.inRange(hls, lower_white, upper_white)
                            # Bitwise-AND mask and original image
                            res = cv.bitwise_and(region_, region_, mask=mask)

                            win, max_val_, max_loc_ = phase_find_best_match(res, [win_mask_720], None)

                        else:
                            region_, x1_, y1_ = submatrix_video(frameV, (495, 190), (495 + 800, 190 + 150))
                            hls = cv.cvtColor(region_, cv.COLOR_BGR2HLS_FULL)

                            # Threshold the HLS image to get only white colors
                            mask = cv.inRange(hls, lower_white, upper_white)
                            # Bitwise-AND mask and original image
                            res = cv.bitwise_and(region_, region_, mask=mask)

                            win, max_val_, max_loc_ = phase_find_best_match(res, [win_mask_1080], None)
                        if visualization_deathORwin:
                            #cv.imshow("winframecrop", res)
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
                            #if only 2 players are left and player wins -> one more kill to the count
                            corrected_win_kills = last_known_valid_kill_count[-1]
                            corrected_player_count = last_known_valid_player_count[-1]

                            if last_known_valid_player_count[-1] == 2:
                                corrected_win_kills = corrected_win_kills + 1

                            #if win not imideatly recognissed & updated count recognised -> +1 to playercount to correct output
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
                    avgLastPosition_icon = (0,0,0)
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

                if video_height == 720:
                    jump, max_val_jump, _ = phase_find_best_match(region, [jump_check_icon720], mask_icon720)
                else:
                    jump, max_val_jump, _ = phase_find_best_match(region, [jump_check_icon1080], mask_icon1080)

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
                if video_height == 720 and FnUseCaseStudy:
                    frameMap, x1, y1 = submatrix_video(frameV, (1053, 25), (1053 + 190, 25 + 190))
                    f = cv.cvtColor(frameMap, cv.COLORSPACE_RGBA)

                elif video_height == 720:
                    frameMap, x1, y1 = submatrix_video(frameV, (1018, 40), (1018 + 185, 40 + 185))
                    f = cv.cvtColor(frameMap, cv.COLORSPACE_RGBA)

                else:
                    frameMap, x1, y1 = submatrix_video(frameV, (1525, 55), (1525 + 277, 55 + 277))
                    f = cv.cvtColor(frameMap, cv.COLORSPACE_RGBA)
                    f = cv.resize(f, None, None, 0.66, 0.66, cv.INTER_CUBIC)

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

                # mr
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
                    n = 250
                    p = lastPositions[-1]
                    img_small, x1, y1 = submatrix(img, p, n)
                    res = cv.matchTemplate(img_small, frame, cv.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                    top_left = (max_loc[0] + x1, max_loc[1] + y1)

                else:
                    res = cv.matchTemplate(img, frame, cv.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
                    top_left = max_loc

                # check if it is on the island or not ...
                if img_mask[int(top_left[1] + w / 2), int(top_left[0] + h / 2)] < 128:
                    omitFrame = True
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
                if video_height == 720:
                    if output_read_num_next_to_icon:
                        print("players:")
                    count_players, count_img = read_num_next_to_icon(frameV, video_height, person_icon720,
                                                                     person_icon_mask720,
                                                                     lastPosition_icon, tool, builder)
                    if output_read_num_next_to_icon:
                        print("kills:")
                    count_player_kills, kills_img = read_num_next_to_icon(frameV, video_height, kills_icon720,
                                                                          person_icon_mask720,
                                                                          lastPosition_icon,
                                                                          tool,
                                                                          builder)

                else:
                    if output_read_num_next_to_icon:
                        print("players:")
                    count_players, count_img = read_num_next_to_icon(frameV, video_height, person_icon1080,
                                                                     person_icon_mask1080,
                                                                     lastPosition_icon, tool, builder)
                    if output_read_num_next_to_icon:
                        print("kills:")
                    count_player_kills, kills_img = read_num_next_to_icon(frameV, video_height, kills_icon1080,
                                                                          person_icon_mask1080,
                                                                          (lastPosition_icon[0] + 5,
                                                                           lastPosition_icon[1]),
                                                                          tool, builder)
                if output_read_num_next_to_icon and visualization_read_num_next_to_icon:
                    cv.imshow("player_count_img", count_img)
                    cv.imshow("kills_count_img", kills_img)
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
                        "{:.0f} {:s} {:2d} {:d} {:f} map position: {:d}x{:d} {:d}".format(frameId, frameTimeStamp, current_phase,
                                                                            last_known_valid_player_count[-1],
                                                                            max_val,
                                                                            top_left[0] + x1,
                                                                            top_left[1] + y1,
                                                                            last_known_valid_kill_count[-1]))

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
                    # cv.imshow("result", heatmap*255/np.max(heatmap))
                    cv.namedWindow('result', cv.WINDOW_NORMAL)
                    cv.resizeWindow('result', 1000, 1000)
                    cv.imshow("result", vis)
                    # cv.imshow("result", cv.resize(vis, (0, 0), fx=0.75, fy=0.75))
                    cv.imshow("frame", frame)
                    cv.waitKey(1)
    cap.release()
    if not waiting4newround:
        write_all(outfile_all, filename, roundId, starttime, frameTimeStamp, -1,
                  last_known_valid_player_count[-1], last_known_valid_kill_count[-1])
        if visualization_general_info:
            print(
                "|######################################ABORTED_GAME####################################|")
