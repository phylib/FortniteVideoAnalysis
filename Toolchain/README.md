# Game Video Stream Analysis Toolchain

The Game Video Stream Analysis Toolchain basically consists of three steps:

1. Video Downloading
2. Data Extraction
3. Data Visualization

## Video Downloading

The video downloading part is done by using the command line tool [youtube-dl](https://github.com/rg3/youtube-dl). More information and scripts for using the tool more efficiently can be found in the **VideoDownloading** subfolder.

## Data Extraction

The data extraction process is automated by a python script. The script combines computer vision tasks of the tools **OpenCV** and **Tesseract**. The source code is based on the work of [1] and can be found along with instructions about how to use it in the **DataExtraction** subfolder.

## Data Visualization

The **DataVisualization** subfolder contains scripts and instructions to generate heat maps and hot spot maps based on movement traces.


[1] P. Moll, M. Lux, S. Theuermann and H. Hellwagner, "A Network Traffic and Player Movement Model to Improve Networking for Competitive Online Games," 2018 16th Annual Workshop on Network and Systems Support for Games (NetGames), Amsterdam, 2018, DOI: 10.1109/NetGames.2018.8463390
