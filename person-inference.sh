#!/bin/bash

if [ "$#" -lt 2 ]
then
  echo "Usage : $0  model_filename  image1 [image2 ...]"
  exit 1
fi


# ==== Pytorch
PTDIR=/home/eric/_tmp/glimpse-clouds/python3-env/bin/activate

echo
echo "Using PyTorch environment in ====> $PTDIR <======="
source $PTDIR

echo -n "Working directory:"
pwd

python3  person_inference.py  $*

#eog $*
