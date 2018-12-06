# Game Video Stream Analysis Toolchain

The Game Video Stream Analysis Toolchain basically consists of three steps:

1. Video Downloading
2. Data Extraction
3. Data Visualization

## Video Downloading

The video downloading part is done by using the command line tool [youtube-dl](https://github.com/rg3/youtube-dl). More information and scripts for using the tool more efficiently can be found in the **VideoDownloading** subfolder.

## Data Extraction

The data extraction process is automated by a python script. The script combines computer vision tasks of the tools **OpenCV** and **Tesseract**. The newest version of the python script can be found on [Bitbucket](https://bitbucket.org/dermotte/gamevideoanalytics/src/fortnitepaper2/). A snapshot of the repository as ZIP-file (Commit: [b464ac0](https://bitbucket.org/dermotte/gamevideoanalytics/commits/b464ac0b3417619d57c57480ec7e5b12d488047c)), which contains all relevant scripts for the analysis and information about how to use it, in the **DataExtraction** subfolder.

## Data Visualization

The **DataVisualization** subfolder contains scripts and instructions to generate heat maps and hot spot maps based on movement traces.
