##################################
#        June 2019               #
##################################


## integrating person attributes into
## ROS-Kinetic/Ubuntu 16.04

# install Python 3.5 for Torch (because native Python 3.4 is too old)
# -> already installed by default

# install Pytorch 0.4.1
$ vim liris_person_attributes/requirements.txt
  # comment line 'ipdb==0.11'
$ virtualenv -p /usr/bin/python3.5 torch-0.4.1-env
$ source torch-0.4.1-env/bin/activate
$ pip install --upgrade pip
$ pip install -r liris_person_attributes/requirements.txt

# install opencv 2.x
apt install libopencv-dev

# install ros video stream (optional)
apt install ros-kinetic-video-stream-opencv

# re-compile cv_bridge for Python 3.5 because standard ROS cv_bridge
# works only with Python 2.7
sudo aptitude install python-catkin-tools python3.5-dev python3-catkin-pkg-modules python3-numpy python3-yaml ros-kinetic-cv-bridge
mkdir catkin_ws_py35
cd catkin_ws_py35
catkin init
catkin config -DPYTHON_EXECUTABLE=path_to/torch-0.4.1-env/bin/python3.5 -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so
catkin config --install
git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv
apt-cache show ros-kinetic-cv-bridge | grep Version
	Version: 1.12.8...
cd src/vision_opencv/
git checkout 1.12.8
vim src/vision_opencv/cv_bridge/CMakeLists.txt
  # replace line
  #   find_package(Boost REQUIRED python3)
  # by
  #   find_package(Boost REQUIRED python-py35)
cd ../../
source path_to/torch-0.4.1-env/bin/activate
source /opt/ros/kinetic/setup.bash
catkin build cv_bridge
source install/setup.bash --extend
