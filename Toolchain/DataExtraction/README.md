## Video Analytics Script

This Script extracts information from Fortnight game-play videos (Made for patch version V.3.1.0, Xbox). It creates trace-files for each match and a meta-file that contains information about the matches.

To use it, you need 'python3' and OpenCV. Learn how to install OpenCV [here](https://milq.github.io/install-opencv-ubuntu-debian/) using [this script](https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh)

The meta-file contains information about:

* Filename the match was taken from.
* Start-time of the match in the video (Start-time referring to when the script start analyzing the match).
* End-time of the match in the video (End-time referring to when the script recognizes death or a win of the player).
* Information if the game was won(1), lost(0) or incorrectly recognized(-1).
* Finale Placement in the match.
* Finale number of kills in the match.

The trace-files each contain information about the match in a 1 second interval:

* Phase recognized in frame
* Position on the map (according to the 1500x1500 map)
* Distance form last position
* Time-stamp of the video
* Placement in current frame
* Kills in current frame

Use the script:

    $ python3 template_match_4.py [param1] [param2] ...

The following params are available:

necessary:

* '-i', '--input': Video file to process.
* '-m', '--metaoutput': Output CSV file for meta information of the rounds.
* '-r', '--roundoutput': Output CSV file for detailed information of a round (trace-files).

optional:

* '-v', '--verbose': Print general processing information in console. (type=bool, default=False)



## Group Size Analytics Script

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
