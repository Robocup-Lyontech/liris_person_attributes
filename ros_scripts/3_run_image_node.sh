#!/bin/bash

SCRIPTABSPATH="$(readlink -f "$0")"
SCRIPTDIR="$(dirname "$SCRIPTABSPATH")"

roslaunch  "$SCRIPTDIR/../launch/el_video_file.launch"


