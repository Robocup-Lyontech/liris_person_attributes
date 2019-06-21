#!/bin/bash

# ==== Pytorch 0.2
PTDIR=/home/chris/pytorch0.3.1/bin/activate

echo
echo "Using PyTorch environment in ====> $PTDIR <======="
source $PTDIR

LOGF="$HOME/LOGS/logfile_jobid_"$SLURM_JOBID".txt"

echo -n "Working directory:"
pwd

echo "All output goes into the logfile: $LOGF"

python person.py > $LOGF 2>&1 


echo "Script terminated."
