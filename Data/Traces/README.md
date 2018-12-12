The __game_overview__-file contains general information about each single game:

* **Video-Filename**, from which the game was taken from.
* **Round-ID** of the game played in the Video.
* **Start-time** of the game in the video.
* **End-time** of the game in the video.
* **Won/Lost**: Information if the game was won(1), lost(0) or incorrectly recognized(-1).
* **Final Placement** in the game.
* **Number of kills** in the game.
* **Play-time** of how long the game was.
* **Corresponding trace file** containing the movement trace of the game.
* **Group-size** Information about how many players(1-4) were in the team or -1 if not recognized).

The __movement_trace__-files, each containing movement information about a single game, sampled in a one second interval:

* **Current Phase** of the game
* **Position** of the player on the 1500x1500px sized map
* **Distance** from last position
* **Time-stamp**
* **Current placement** at the current time
* **Current number of kills**

Please note that the data also contains information about games which were extracted with errors (`won=-1`). Those data points have to be ignored when doing analysis.

Please also note that videos files with a name starting with `FN[xx]` result from the user study. All other games result from experienced Fortnite streamers, as explained in the paper.
