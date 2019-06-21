#!/bin/bash

TIMESTAMP="$(date +%y%m%d-%H%M%S)"
WORKINGDIR="jperson_elo.$TIMESTAMP"

if [ "$HOSTNICKNAME" = "Oggy" ]
then
  COM="mkdir "$WORKINGDIR" && cd "$WORKINGDIR" && sbatch --gres=gpu:1 --mem=14000 ../jperson_elo.sh $*"
else
  COM="mkdir "$WORKINGDIR" && cd "$WORKINGDIR" && ../jperson_elo.sh $*  2>&1 | tee jperson_elo.output.$TIMESTAMP"
fi

echo $COM
eval $COM
