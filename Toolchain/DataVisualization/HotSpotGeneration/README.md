# Hotspot generation

Based on heat maps, \textit{hot spot maps} can be extracted. A hot spot is defined as region, where the activity was above a certain threshold. A hot spot map is created by binarizing the heat map and by reducing to small spots by eroding and dilating the resulting image.

This process is automated by a bash-script (`hotspot.sh`). Basically, the script uses `imagemagick`'s `convert` tool for binarizing the input heatmap. Afterwards the resulting spots are eroded and dilated. Finally, the hot spot map is stored with different hot spot colors.

Sample usage:

    ./hotspot.sh [input/heatmap/filename.png]
