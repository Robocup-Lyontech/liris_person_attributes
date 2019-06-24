#!/bin/bash
# installing person attributes into
# ROS-Kinetic/Ubuntu 16.04
#
# Prerequisites:
# - Python 3.5 for Torch, already installed by default
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

# install Pytorch 0.4.1
virtualenv  -p /usr/bin/python3.5 torch-0.4.1-env
source torch-0.4.1-env/bin/activate
pip install --upgrade pip
pip install -r liris_person_attributes/requirements.txt

# install extra packages in virtualenv to interact with ROS
pip install PyYAML
pip install rospkg
pip install opencv-python

# re-compile cv_bridge for Python 3.5 because standard ROS cv_bridge works only with Python 2.7
mkdir catkin_ws_py35
cd catkin_ws_py35
catkin init
catkin config -DPYTHON_EXECUTABLE=../torch-0.4.1-env/bin/python3.5 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin config --install
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
#apt-cache show ros-kinetic-cv-bridge | grep Version
#	Version: 1.12.8...
(cd src/vision_opencv && git checkout 1.12.8)
cp src/vision_opencv/cv_bridge/CMakeLists.txt src/vision_opencv/cv_bridge/CMakeLists.txt.bak
# fix cv_bridge/CMakeLists.txt
#vim src/vision_opencv/cv_bridge/CMakeLists.txt
#  # replace line
#  #   find_package(Boost REQUIRED python3)
#  # by
#  #   find_package(Boost REQUIRED python-py35)
sed 's/Boost REQUIRED python3/Boost REQUIRED python-py35/' < src/vision_opencv/cv_bridge/CMakeLists.txt.bak  > src/vision_opencv/cv_bridge/CMakeLists.txt
source /opt/ros/kinetic/setup.bash
catkin build cv_bridge
source install/setup.bash --extend

