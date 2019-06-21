#!/bin/bash

SCRIPTABSPATH="$(readlink -f "$0")"
SCRIPTDIR="$(dirname "$SCRIPTABSPATH")"

set -e

#source ~/_prog/robocup2018_ros/pytorch-python27-env/bin/activate
source "$SCRIPTDIR/../../torch-0.4.1-env/bin/activate"

source ~/catkin_ws_py35/devel/setup.bash

#rosrun  test_el  person_attributes_ros.py
python3 person_attributes_ros.py
