#!/bin/bash

date

# ==== Pytorch
PTDIR=/home/eric/_tmp/glimpse-clouds/python3-env/bin/activate

echo
echo "Using PyTorch environment in ====> $PTDIR <======="
source $PTDIR

#LOGF="$HOME/LOGS/logfile_jobid_"$SLURM_JOBID".txt"

echo -n "Working directory:"
pwd

#echo "All output goes into the logfile: $LOGF"

if [ -n "$SLURM_SUBMIT_DIR" ]
then
  SCRIPTDIR="$(readlink -f "$SLURM_SUBMIT_DIR/..")"
else
  SCRIPTPATH="$(readlink -f "$0")"
  SCRIPTDIR="$(dirname "$SCRIPTPATH")"
fi

#python person.py > $LOGF 2>&1 
python3  "$SCRIPTDIR"/person.py  $*


echo "Script terminated."

date
