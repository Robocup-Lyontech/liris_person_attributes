#!/bin/bash
# installing person attributes on
# ROS-Kinetic/Ubuntu 16.04
#
# Prerequisites:
# - Python 2.7
# - sudo apt install libopencv-dev
# - sudo apt install ros-kinetic-video-stream-opencv
# - sudo apt install python-catkin-tools python3.5-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge
#
# Installation in current dir
#

set -e

# check dependencies
# install opencv 2.x
#sudo apt install libopencv-dev
# install ros video stream (optional)
#sudo apt install ros-kinetic-video-stream-opencv
# cv_bridge recompiling stuff
#sudo apt install python-catkin-tools python3.5-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge

# get person attributes code
git clone https://github.com/Robocup-Lyontech/liris_person_attributes.git

# install Pytorch 0.4.1 in Python 2.7 env
virtualenv  torch-0.4.1-env
source torch-0.4.1-env/bin/activate
pip install --upgrade pip
pip install -r liris_person_attributes/requirements27.txt

# install extra packages in virtualenv to interact with ROS
pip install PyYAML
pip install rospkg
pip install opencv-python
deactivate

# create launch scripts

# script 1: roscore
echo " #!/bin/bash

roscore
" > 1_run_roscore.sh

# script 2: person_attributes
ENVFULLPATH="$(readlink -f ./torch-0.4.1-env)"
echo "#!/bin/bash

SCRIPTABSPATH=\"$(readlink -f \"$0\")\"
SCRIPTDIR=\"$(dirname \"$SCRIPTABSPATH\")\"

set -e

source \"$ENVFULLPATH/bin/activate\"
source ~/catkin_ws/devel/setup.bash

cd \"$SCRIPTDIR/liris_person_attributes\"
python ./person_attributes_ros.py
" > 2_run_person_attributes_ros.sh

# script 3: image node
echo "#!/bin/bash

SCRIPTABSPATH=\"$(readlink -f \"$0\")\"
SCRIPTDIR=\"$(dirname \"$SCRIPTABSPATH\")\"

roslaunch  \"$SCRIPTDIR/../launch/el_video_file.launch\"
" > 3_run_ros_image_node.sh


