# /bin/bash

INPUT=$1
NAME=${INPUT%.png}
TARGET=${NAME}Hotspot.png
BLUE=#66B2FF
GREEN=#B2FF66
ORANGE=#FF9933
BTARGET=${NAME}HotspotBlue.png
GTARGET=${NAME}HotspotGreen.png
OTARGET=${NAME}HotspotOrange.png

convert $INPUT -threshold 50% $TARGET
convert $TARGET -morphology Erode Octagon:10 $TARGET
convert $TARGET -morphology Dilate Octagon:10 $TARGET
convert $TARGET -fill $BLUE -opaque white $BTARGET
convert $TARGET -fill $GREEN -opaque white $GTARGET
convert $TARGET -fill $ORANGE -opaque white $OTARGET
