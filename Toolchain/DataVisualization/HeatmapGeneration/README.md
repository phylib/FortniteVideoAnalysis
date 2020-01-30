## Heatmap Generation Script

This script generates the heatmaps based on trace-files and the meta-file. It generates up to 4 heatmaps depending on which parameters are given. The "all" and "kills" map are always generated, the start-heatmap and end-heatmap map are only generated if no start-time and no end-time are given.
Start-heatmap uses the first seconds of the tracings where the player landed. End-heatmap the last seconds.
The kills-heatmap marks the places from which the player has killed another player from. (The Kills marked are also constrained by the min & max distance parameter. In order to receive an kills-heatmap with all the kills you have to set '-n 0' (min) and not set a 'x-'(max) as shown in an example below.)

To use it, you need 'python3' and a number of python packages, such as  “heatmappy.py”. Python will tell you which packages are messing, install them with pip3.

Changes to the original meta-file are adding 2 columns "playtime" and "tracename" after the "kills" column.
* playtime: Endtime - starttime ; for easier filtering
* tracename: "text to column" Function on the "video" column (spliting by "/" and ".") and using concat to recreate the tracefile name; Example: '=CONCAT("round-",<text to column cut>,"_",<roundId>,".csv"')

Use the script:

    $ python3 normalized_heatmap.py [param1] [param2] ...

The following params are available:

necessary:

* '-d', '--inputfiles': Folder with the trace files as csv.
* '-i', '--inputlist': File with which traces should be used.
* '-o', '--output': Output folder and files name.

optional:

* '-t', '--time': Set the time given in seconds for the start and end heatmap (type=int, default=5)
* '-x', '--max': Max distance a player moves per sec (type=float, default=22)
* '-n', '--min': Min distance a player moves per sec (lowest value = 0) (type=float, default=1)
* '-w', '--won': Filter if match was won or lost (type=bool, default=None)
* '-g', '--groupsize', Generate the heatmap with the given group-size (type=int, default=None)
* '-s', '--starttime', Generate the heatmap from the time given in seconds (type=int, default=0)
* '-e', '--endtime', Generate the heatmap till the time given in seconds. The Value 0 generates the heatmap till the end of the round. (type=int, default=0)


Examples:

Generate a heatmap of all winners and mark only if the movement was higher than .5 distance (kills Heat map is also constrained by movent speed which results in missing kills)

    $ python3 normalized_heatmap.py -d ./files/ -i ./files/meta-test_improved.csv -o all_wins -n .5 -w 1

To generate a heatmap of all kills that all winners made set the distance to 0

    $ python3 normalized_heatmap.py -d ./files/ -i ./files/meta-test_improved.csv -o all_wins -n 0 -w 1  True

Using a filtered version of Meta-test_improved.csv to generate heatmaps of the places the top 10 players moved around the least

    $ python3 normalized_heatmap.py -d ./files/ -i ./files/meta-test_top10.csv -o top_10 -n .0 -m 1.5

Generate a heatmap of the time between 5 and 10 minutes

    $ python3 normalized_heatmap.py -d ./files/ -i ./files/meta-test_improved.csv -o min5-10 -s 300 -e 600
