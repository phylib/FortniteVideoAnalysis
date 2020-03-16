
import numpy
import numpy as np
import time
import itertools
from heatmappy import Heatmapper
from PIL import Image
import argparse
import os
import datetime



class Fields:
    phase, x_pos, y_pos, distance, video_time, place, kills = range(0, 7)

#class Fields_file_list:
#   tracename, starttime, endtime, won, place, kills, playtime = range(0, 7)

class Fields_file_list:
    video, roundId, starttime, endtime, won, place, kills, playtime, tracename, groupsize = range(0, 10)


heatmap_start = numpy.zeros((1500, 1500))
heatmap_all = numpy.zeros((1500, 1500))
heatmap_end = numpy.zeros((1500, 1500))
heatmap_kills = numpy.zeros((1500, 1500))
points_start = []
points_all = []
points_end = []
points_kills = []
intervalSE = None
dist_max = None
dist_min = None
landed_time_check = 10


def find_first(list,number):

    for i in range(0,len(list)):
        if list[i][0] == number:
            return list[i][1]


def get_correct_case_filename(filename):
    folder = "/".join(filename.split("/")[:-1])
    fileonly_name = filename.split("/")[-1]
    files = os.listdir(folder)
    for file in files:
        if file.lower() == fileonly_name.lower():
            fileonly_name = file
            break
    return folder + "/" + fileonly_name

def ingest_file_content(file_path):
    temp = 0
    #file_path = file_path[0:-1]
    last_pos_list = []
    dist = []
    landed = False
    last_value_false = 0
    last_corr_pos = (-1, -1)

    #kills variables:
    #
    last_kill = 0
    last_kills_count = []#[(0, (0, 0))] * 30
    last_place_count = 100

    round_start_time = None
    file_path = get_correct_case_filename(file_path)

    with open(file_path, 'r') as file:
        for line in file.readlines()[1:]:  # skip over header

            parts = line.split(',')

            # skip first
            if round_start_time is None:
                round_start_time = parts[Fields.video_time].split(":")
                round_start_sec = int(round_start_time[0])*60*60+int(round_start_time[1])*60+int(round_start_time[2])

            round_time = parts[Fields.video_time].split(":")
            time_sec = int(round_time[0])*60*60+int(round_time[1])*60+int(round_time[2])
            round_time_sec = time_sec-round_start_sec
            #print(
            #    "video start time: {:d} video time: {:d} game time: {:d}".format(round_start_sec,time_sec,round_time_sec))

            # find the first 10 values under 3.606 to find estimated landing spot
            if not landed:
                # add location to list if value is slow enough to not be flight else reset list
                if dist.__len__() < landed_time_check:
                    if 0 <= float(parts[Fields.distance]) <= 3.606:
                        dist.append((float(parts[Fields.distance]), int(parts[Fields.x_pos]), int(parts[Fields.y_pos]), round_time_sec))
                    else:
                        dist = []
                # if list has enough entries add them to the final heatmaps
                if dist.__len__() >= landed_time_check:
                    landed = True
                    for i in range(dist.__len__()):
                        # check if position fits with given parameters before adding
                        if dist[i][0] >= dist_max or dist[i][0] <= dist_min:
                            # landing spots (first locations)
                            if i <= intervalSE and not timeslice:
                                heatmap_start[dist[i][1]][dist[i][2]] += 1
                                temp += 1
                            # all
                            if (start_time_constrain <= dist[i][3] <= end_time_constrain) or \
                                    (start_time_constrain <= dist[i][3] and end_time_constrain == 0) or \
                                    (round_time_sec <= dist[i][3] and start_time_constrain == 0):
                                heatmap_all[dist[i][1]][dist[i][2]] += 1

                            # end
                            if not timeslice:
                                last_pos_list.append((dist[i][1], dist[i][2]))
                        last_corr_pos = (dist[-1][1], dist[-1][2])
                    dist = []

            else:
                if last_value_false > 0:
                    # if landed and the last value was false calculate the distance according to last correct val and new position
                    distance = numpy.math.sqrt(
                        (last_corr_pos[0] - int(parts[Fields.x_pos])) ** 2 +
                        (last_corr_pos[1] - int(parts[Fields.y_pos])) ** 2)
                    #print("calculated dist:\t {:f}".format(distance))

                else:
                    distance = float(parts[Fields.distance])
                    #print("Given dist:\t {:f}".format(distance))

                # check if position fits with given parameters
                # (+ last_value_false * distance if map is opend & player is moving)
                if distance >= (dist_max + last_value_false * dist_max) or distance <= dist_min:
                    if distance >= (3.606 + last_value_false * 3.606) or distance < 0:
                        # if last position exists keep last position
                        if last_pos_list.__len__() > 0:
                            last_value_false += 1
                            last_corr_pos = (last_pos_list[-1][0], last_pos_list[-1][1])

                # if correct location was found and distance is normal
                else:
                    last_value_false = 0

                    # start (landing spots/first locations)
                    if not timeslice:
                        if temp <= intervalSE:
                            heatmap_start[int(parts[Fields.x_pos])][int(parts[Fields.y_pos])] += 1
                            temp += 1

                    # all
                    if (start_time_constrain <= round_time_sec <= end_time_constrain) or \
                        (start_time_constrain <= round_time_sec and end_time_constrain == 0) or \
                            (round_time_sec <= end_time_constrain and start_time_constrain == 0):
                        heatmap_all[int(parts[Fields.x_pos])][int(parts[Fields.y_pos])] += 1

                    # end
                    if not timeslice:

                        last_pos_list.append((int(parts[Fields.x_pos]), int(parts[Fields.y_pos])))
                        last_pos_list = last_pos_list[-intervalSE:]
                        last_corr_pos = (last_pos_list[-1][0], last_pos_list[-1][1])

                    # kills
                    # add kill when...
                    cur_place_count = int(parts[Fields.place])
                    cur_kills_count = int(parts[Fields.kills])

                    last_kills_count.append((cur_kills_count, (int(parts[Fields.x_pos]), int(parts[Fields.y_pos]), round_time_sec)))
                    #print(last_kills_count)
                    #print(last_place_count, cur_place_count)
                    if len(last_kills_count) > 30:
                        last_kills_count = last_kills_count[1:]

                    # clear case
                    if cur_kills_count == last_kill+1 and cur_place_count == last_place_count-1:
                        if (start_time_constrain <= round_time_sec <= end_time_constrain) or \
                                (start_time_constrain <= round_time_sec and end_time_constrain == 0) or \
                                (round_time_sec <= end_time_constrain and start_time_constrain == 0):
                            heatmap_kills[int(parts[Fields.x_pos])][int(parts[Fields.y_pos])] += 1
                        last_kill = cur_kills_count
                        last_kills_count = last_kills_count[len(last_kills_count)-1:]
                        #print("clear case! kills: {:d} {:s}".format(last_kill, parts[Fields.video_time]))

                    # check if kills went up and place went down reasonably
                    elif cur_kills_count > last_kill and cur_kills_count + cur_place_count <= 100 and\
                            cur_place_count < last_place_count: #last_place_count - 15 <= \
                        # majority vote

                        k = len(last_kills_count)
                        buffer = 8
                        voter = {}
                        if k >= 16:
                            for i in range(0, k):
                                if last_kills_count[-1 - i][0] not in voter:
                                    voter[last_kills_count[-1 - i][0]] = 1
                                else:
                                    voter[last_kills_count[-1 - i][0]] += 1

                            voter_sorted = sorted(voter, key=voter.get, reverse=True)
                            new_number = voter_sorted[0]

                            #print("1st voter.get({:d}) = {:d}".format(new_number,
                            #                                      voter.get(new_number)))

                            if not last_kill < new_number <= last_kill + buffer and len(voter_sorted) > 1:
                                second_place_new_number = voter_sorted[1]
                                #print("2nd voter.get({:d}) = {:d}".format(second_place_new_number,
                                #                                      voter.get(second_place_new_number)))
                                if voter.get(second_place_new_number) > 16:
                                    # false reading of 4 instead of 1 filtered out
                                    if last_kill < second_place_new_number <= last_kill + buffer and not last_kill == 1\
                                            and second_place_new_number == 4:
                                        new_number = second_place_new_number

                            if new_number > 0 and new_number > last_kill:
                                last_kill = new_number
                                new_position = find_first(last_kills_count, new_number)
                                #print("new position: {:d} x {:d}".format(new_position[0], new_position[1]))
                                if (start_time_constrain <= new_position[2] <= end_time_constrain) or \
                                        (start_time_constrain <= new_position[2] and end_time_constrain == 0) or \
                                        (new_position[2] <= end_time_constrain and start_time_constrain == 0):
                                    heatmap_kills[new_position[0]][new_position[1]] += 1
                                #print(sorted(voter, key=voter.get, reverse=True), voter, last_kills_count[-len(last_kills_count):-1])

                    #update last position
                    last_place_count = cur_place_count

    for k in range(0, last_pos_list.__len__()):
        #print(last_pos_list[k])
        heatmap_end[last_pos_list[k][0]][last_pos_list[k][1]] += 1


def newHeatmapper(point_diameter=50):
    heatmapper_cutout = Heatmapper(point_diameter=point_diameter,  # the size of each po    int to be drawn
                                   point_strength=0.005,  # the strength, between 0 and 1, of each point to be drawn
                                   opacity=1,  # the opacity of the heatmap layer
                                   colours='reveal',  # 'default' or 'reveal'
                                   # OR a matplotlib LinearSegmentedColorMap object
                                   # OR the path to a horizontal scale image
                                   grey_heatmapper='PIL'  # The object responsible for drawing the points
                                   # Pillow used by default, 'PySide' option available if installed
                                   )

    heatmapper = Heatmapper(point_diameter=point_diameter,  # the size of each point to be drawn
                            point_strength=0.02,  # the strength, between 0 and 1, of each point to be drawn
                            opacity=1,  # the opacity of the heatmap layer
                            colours='default',  # 'default' or 'reveal'
                            # OR a matplotlib LinearSegmentedColorMap object
                            # OR the path to a horizontal scale image
                            grey_heatmapper='PIL'  # The object responsible for drawing the points
                            # Pillow used by default, 'PySide' option available if installed
                            )
    return heatmapper_cutout, heatmapper


parser = argparse.ArgumentParser(description='Script to generate Fortnight Heatmaps.')
parser.add_argument("-d", "--inputfiles", help="Folder with the trace files.")
parser.add_argument("-i", "--inputlist", help="File with which traces should be used.")
parser.add_argument("-o", "--output", help="Output folder and files name.", default="heatmaps/")
parser.add_argument("-t", "--time", help="Start and end interval that's mapped (in seconds)", type=int, default=5)
parser.add_argument("-x", "--max", help="Max distance a player moves per sec", type=float, default=3.606)
parser.add_argument("-n", "--min", help="Min distance a player moves per sec (lowest value = 0)", type=float, default=1)
parser.add_argument("-w", "--won", help="filter if match was won or lost", type=bool, default=None)
#parser.add_argument("-k", "--killsmap", help="only generate the heatmap with the kills", type=bool, default=False)
parser.add_argument("-g", "--groupsize", help="only generate the heatmap with the given groupsize", type=int, default=None)
parser.add_argument("-s", "--starttime", help="only generate the heatmap from given start time (given in seconds)", type=int, default=0)
parser.add_argument("-e", "--endtime", help="only generate the heatmap till given end time (given in seconds)", type=int, default=0)


args = parser.parse_args()
folder = args.inputfiles
file_list = args.inputlist
output = args.output
intervalSE = args.time
dist_max = args.max
dist_min = args.min
won_filter = args.won
# = args.killsmap
groupsize = args.groupsize
start_time_constrain = args.starttime
end_time_constrain = args.endtime


filecount = 0
if dist_min < 0:
    dist_min = 0
if dist_max < 0:
    dist_max = 0
if groupsize is not None:
    if groupsize < 1 or groupsize > 4:
        groupsize = None

if start_time_constrain != 0 or end_time_constrain != 0:
    timeslice = True
else:
    timeslice = False

with open(file_list, 'r') as files:
    for line in files.readlines()[1:]:  # skip over header
        parts = line.split(',')   # \t or ,
        if parts[Fields_file_list.won] == '-1': # skip aborted rounds
            continue
        start = time.time()
        if won_filter is None and groupsize is None:
            #print(Fields_file_list.tracename)
            ingest_file_content(folder + parts[Fields_file_list.tracename])

        elif won_filter is None and groupsize == int(parts[Fields_file_list.groupsize]):
            #print(Fields_file_list.tracename)
            ingest_file_content(folder + parts[Fields_file_list.tracename])

        elif won_filter == int(parts[Fields_file_list.won]) and groupsize is None:
            #print(Fields_file_list.tracename)
            ingest_file_content(folder + parts[Fields_file_list.tracename])

        elif won_filter == int(parts[Fields_file_list.won]) and groupsize == int(parts[Fields_file_list.groupsize]):
            #print(Fields_file_list.tracename)
            ingest_file_content(folder + parts[Fields_file_list.tracename])

        end = time.time()
        filecount += 1
        print("Filecount: {:2d} ".format(filecount) + " processed " + folder + parts[Fields_file_list.tracename] + " in " + str(end - start) + 's')

map_path = 'map.png'
map_img = Image.open(map_path)
white_path = 'white.png'
white_img = Image.open(white_path)

heatmapper_cutout_start, heatmapper_start = newHeatmapper(point_diameter=100)
heatmapper_cutout_all, heatmapper_all = newHeatmapper()
heatmapper_cutout_end, heatmapper_end = newHeatmapper()

outpath = output
#outpath = 'heatmaps/' + output +'_'+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '/'
os.makedirs(outpath, exist_ok=True)

# normalize count before handing it over to heatmappy

heatmap_start /= (heatmap_start.max() / 100)
heatmap_all /= (heatmap_all.max() / 100)
heatmap_end /= (heatmap_end.max() / 100)

#print(heatmap_start.max())
if heatmap_start.max() > 0:

    for j in range(len(heatmap_start)):
        for i in range(len(heatmap_start[0])):
            if heatmap_start[i][j] == 0: continue
            points_start += list(itertools.repeat((i, j), int(round(heatmap_start[i][j]))))

    heatmap_start = heatmapper_start.heatmap_on_img(points_start, map_img)
    heatmap_start.save(outpath + output + '_heatmap_norm_start.png')

    heatmap_cutout_start = heatmapper_cutout_start.heatmap_on_img(points_start, white_img)
    heatmap_cutout_start.save(outpath + output + '_heatmap_cutout_norm_start.png')

if heatmap_all.max() > 0:
    for j in range(len(heatmap_all)):
        for i in range(len(heatmap_all[0])):
            if heatmap_all[i][j] == 0: continue
            points_all += list(itertools.repeat((i, j), int(round(heatmap_all[i][j]))))

    heatmap_all = heatmapper_all.heatmap_on_img(points_all, map_img)
    heatmap_all.save(outpath + output + '_heatmap_norm_all.png')

    heatmap_cutout_all = heatmapper_cutout_all.heatmap_on_img(points_all, white_img)
    heatmap_cutout_all.save(outpath + output + '_heatmap_cutout_norm_all.png')

if heatmap_end.max() > 0:

    for j in range(len(heatmap_end)):
        for i in range(len(heatmap_end[0])):
            if heatmap_end[i][j] == 0: continue
            points_end += list(itertools.repeat((i, j), int(round(heatmap_end[i][j]))))

    heatmap_end = heatmapper_end.heatmap_on_img(points_end, map_img)
    heatmap_end.save(outpath + output + '_heatmap_norm_end.png')

    heatmap_cutout_end = heatmapper_cutout_end.heatmap_on_img(points_end, white_img)
    heatmap_cutout_end.save(outpath + output + '_heatmap_cutout_end.png')

# check if kills were made
if heatmap_kills.max() > 0:
    heatmap_kills /= (heatmap_kills.max() / 100)

    for j in range(len(heatmap_kills)):
        for i in range(len(heatmap_kills[0])):
            if heatmap_kills[i][j] == 0: continue
            points_kills += list(itertools.repeat((i, j), int(round(heatmap_kills[i][j]))))

    heatmapper_cutout_kills, heatmapper_kills = newHeatmapper()

    heatmap_kills = heatmapper_kills.heatmap_on_img(points_kills, map_img)
    heatmap_kills.save(outpath + output + '_heatmap_norm_kills.png')

    heatmap_cutout_kills = heatmapper_cutout_kills.heatmap_on_img(points_kills, white_img)
    heatmap_cutout_kills.save(outpath + output + '_heatmap_cutout_kills.png')


