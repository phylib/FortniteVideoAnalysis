## Video Analytics Script

This Script extracts information from Fortnight game-play videos (Streamed from 11.29.2019 to 12.12.2019; Game version V.11.2 and V.11.2.1). It creates trace-files for each match and a meta-file that contains information about the matches.

To use it, you need 'python3' and OpenCV. Learn how to install OpenCV [here](https://milq.github.io/install-opencv-ubuntu-debian/) using [this script](https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh)

The meta-file contains information about:

* Filename the match was taken from.
* Start-time of the match in the video (Start-time referring to when the script start analyzing the match).
* End-time of the match in the video (End-time referring to when the script recognizes death or a win of the player).
* Information if the game was won(1), lost(0) or incorrectly recognized(-1).
* Finale Placement in the match (not Valid in this version).
* Finale number of kills in the match (not Valid in this version).

The trace-files each contain information about the match in a 1 second interval:

* Phase recognized in frame
* Position on the map (according to the 1500x1500 map)
* Distance form last position
* Time-stamp of the video
* If player is gliding down during the jumphase
* Confidence Value of Map and Phase detection in current frame
* If Frame is to be omited in Heatmapgeneration (Position somewhere in the water)

Use the script:

    $ python3 template_match_4_scaled.py [param1] [param2] ...
-o /home/itec-linux/Desktop -v False -t True
The following params are available:btw

necessary:

* '-i', '--input': Video file to process.
* '-o', '--outpath': Path for output

optional:

* '-v', '--verbose': Print general processing information in console. (type=bool, default=False)
* '-t', '--testrun': Debug test run mode for single videos, output is saved with timestamp. (type=bool, default=False)



## Group Size Analytics Script (Not used in this version)


Based on a slightly modified meta file*), the round trace files & the videos "get_player_group_size.py" creates a new meta-file including a column with the group size of the rounds.
Values of the column 1,...,4 normal group sizes, -1 no result found, Error if video was not found.

*) "playtime" & "tracename" columns were added via LibreOffice based on the given values.

necessary:

* '-i', '--inputlist': Meta file with the traces that should be used.
* '-v', '--videos': Location of the Videos.
* '-t', '--traces': Location of the trace files.

optional:

* '-o', '--output': Output file name.
* '-p', '--print': Print output if True. (type=bool, default=False)
