import argparse
import os
import datetime
import cv2 as cv
import math
import numpy as np


class Fields:
    phase, x_pos, y_pos, distance, video_time, place, kills = range(0, 7)


class Fields_file_list:
    video, roundId, starttime, endtime, won, place, kills, playtime, tracename = range(0, 9)


def ingest_file_content(video_path, file_path):
    file_path = file_path[0:-1]
    counter_sec = 0
    res_arr = []
    cap = cv.VideoCapture(video_path)
    frameRate = math.floor(cap.get(cv.CAP_PROP_FPS))

    ### Set variable for video size (720p or 1080p)to check while analysing
    video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    ret = True
    # ret, frameV = cap.read()

    if print:
        print("analysing:\n\t" + file_path + "\nwith video:\n\t" + video_path)

    # read round trace file
    with open(file_path, 'r') as file:
        for line in file.readlines()[1:]:  # skip over header
            # check only 2 seconds
            if counter_sec > 1:
                break

            # get information from round csv row
            parts = line.split(',')

            # claculate frameid based of timestamp
            timetemp = parts[Fields.video_time].split(':')
            frameid = ((int(timetemp[0]) * 60 + int(timetemp[1])) * 60 + int(timetemp[2])) * 1000
            # print(frameid, timetemp, parts[Fields.phase])

            # set video cap to timestamp of the trace
            cap.set(cv.CAP_PROP_POS_MSEC, frameid)

            if parts[Fields.phase] == '0' or parts[Fields.phase] == '1':
                counter_sec += 1
                counter_frame = 0
                # go through all frames of the second and crop according to video size
                while cap.isOpened() and ret and counter_frame <= frameRate:
                    counter_frame += 1
                    ret, frameV = cap.read()
                    video_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float

                    if video_height == 720:
                        region, x1, y1 = submatrix_video(frameV, (65, 65), (65 + 190, 65 + 160))
                        # cv.imshow("crop720", region)
                        # cv.waitKey(100)

                    else:
                        region, x1, y1 = submatrix_video(frameV, (98, 98), (98 + 285, 98 + 240))
                        # cv.imshow("crop1080", region)
                        # cv.waitKey(100)
                    # get groupsize and add to list
                    res_arr.append(analyse_image(region))

    # remove when no player was found
    while res_arr.count(0) > 0:
        res_arr.remove(0)

    # pick most often found group_size
    group_size = -1
    groupsize_count = 0
    for i in range(1, 5):
        if res_arr.count(i) > groupsize_count:
            group_size = i
            groupsize_count = res_arr.count(i)
    # print(res_arr)
    # print(group_size)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    return group_size


def submatrix_video(matrix, p1=(1000, 190), p2=(1200, 260)):
    return matrix[p1[1]:p2[1], p1[0]:p2[0]], p1[0], p1[1]


def analyse_image(image):
    playercount = 0

    # define the list of boundaries
    boundaries = [
        ([20, 202, 78], [123, 255, 138]),
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries
        mask = cv.inRange(image, lower, upper)
        # and apply the mask
        # output = cv.bitwise_and(image, image, mask=mask)
        # masked_img = np.hstack([image, output])
        # print(masked_img.shape)

        # crop each players healthbar and check if exists
        for i in range(4):
            temp, _, _ = submatrix_video(mask, (0, i * mask.shape[0] / 4), (mask.shape[1], (i + 1) * mask.shape[0] / 4))
            # print(mask.shape)
            # print(temp.shape)
            frame_max_size = temp.shape[0] * temp.shape[1]
            unique, counts = np.unique(temp, return_counts=True)
            res = dict(zip(unique, counts))
            # print(res.get(0), res.get(255))
            # print(res)

            # if no healthbar found
            if res.get(255) == None:
                break
            # if healthbar color found check if size makes sense
            else:
                black_perc = 100 / frame_max_size * res.get(0)
                white_perc = 100 / frame_max_size * res.get(255)
                # ~15% is max healthbar takes up of crop
                if 16 > white_perc > 8:
                    playercount += 1

            # show the images
            # cv.imshow("images", temp)
            # cv.waitKey(100)
    return playercount


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script analyse group size of a round.')
    parser.add_argument("-i", "--inputlist", help="File with which traces should be used.")
    parser.add_argument("-v", "--videos", help="Location of the Videos.", type=str)
    parser.add_argument("-t", "--traces", help="Location of the traces.")
    parser.add_argument("-o", "--output", help="Output file name.")
    parser.add_argument("-p", "--print", help="Print output if True.", type=bool, default=False)

    args = parser.parse_args()
    file_list = args.inputlist
    vids_path = args.videos
    traces_path = args.traces
    output = args.output
    print = args.print

    if output is None:
        output = "Results_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
        output = 'output/' + output
        # os.makedirs(outpath, exist_ok=True)

    if not os.path.isfile(output):
        outfile_all = open(output, "a")
        outfile_all.write("video,roundId,starttime,endtime,won,place,kills,playtime,tracename,groupsize\n")
        outfile_all.close()
    else:
        outfile_all = open(output, "r")
        outfile_all.close()

    with open(file_list, 'r') as files:
        for line in files.readlines()[1:]:  # skip over header
            parts = line.split(',')  # \ŧ or ,

            # print(traces_path + parts[Fields_file_list.tracename])
            # video_path = vids_path + parts[Fields_file_list.video][2:]
            video_path = ''
            video_path = os.path.join(video_path, vids_path + parts[Fields_file_list.video][2:])

            # Usecases were always solo
            if video_path.__contains__("Fn"):
                outfile = open(output, "a")
                outfile.write(line[:-1] + "," + str(1) + "\n")
                outfile.close()

            # aborted rounds
            elif parts[Fields_file_list.won] == '-1':
                outfile = open(output, "a")
                outfile.write(line[:-1] + "," + "-1" + "\n")
                outfile.close()

            # video found for analysing
            elif os.path.isfile(video_path):
                # print(video_path)
                # print("video found")

                group_size = ingest_file_content(video_path, traces_path + parts[Fields_file_list.tracename])

                outfile = open(output, "a")
                outfile.write(line[:-1] + "," + str(group_size) + "\n")
                outfile.close()

            # video not found
            else:
                # print(video_path)
                # print("video not found")
                outfile = open(output, "a")
                outfile.write(line[:-1] + "," + "Error" + "\n")
                outfile.close()
