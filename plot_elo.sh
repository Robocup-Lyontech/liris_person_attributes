#!/bin/bash

if [ $# -ne 1 ]
then
	echo "usage: $0  <log_file>"
	exit 1
fi

LOGF="$1"

if [ ! -f "$LOGF" ]
then
	echo "File does not exit: $LOGF"
	exit 1
fi

TMPF="/tmp/plot_elo_data.tmp"

#
# Display global loss, training error, validation error
#

grep ^STAT "$LOGF" | sed 's/^STAT//' > "$TMPF"

#cat "$TMPF"

echo "
set title \"Global\"
set style line 1 lt 1 lw 2 pt 1 ps 0.5;
set style line 2 lt 3 lw 2 pt 1 ps 0.5;
set style line 3 lt 7 lw 1 pt 1 ps 0.5;
set style line 4 lt 4 lw 2 pt 4 ps 0.5;
set style line 5 lt 5 lw 2 pt 4 ps 0.5;
set style line 6 lt 2 lw 2 pt 2 ps 0.5;
set style line 7 lt 6 lw 2 pt 4 ps 0.5;
set style line 8 lt 8 lw 2 pt 4 ps 0.5;
set style line 20 lt 7 lw 2 pt 4 ps 0.5;
set xlabel \"Batch number\"
set ylabel \"Classif-error\"
#set yrange [0:110]
set autoscale y
set y2label \"Loss\"
set autoscale y2
set y2tics 
set logscale y2
plot \"$TMPF\" using 1:3 title \"Train error\" with lines ls 2, \
     \"$TMPF\" using 1:4 title \"Valid error\" with lines ls 1, \
     \"$TMPF\" using 1:2 title \"Train loss\" with lines ls 3 axes x1y2
" | gnuplot -persist


head "$LOGF"


#
# Display per-attribute training error and validation error
#

# frequency of attribute with value 1 in train database
# sorted by increasing distance to 50%
ATTR_TRAIN_FREQ="up-Black 52.1%
Age31-45 56.0%
lb-LongTrousers 56.1%
Age17-30 43.0%
low-Black 60.9%
shoes-Black 60.0%
faceBack 33.4%
faceFront 31.8%
ub-Jacket 31.5%
Female 30.7%
shoes-Leather 30.4%
attach-Other 29.5%
low-Blue 28.1%
lb-Jeans 26.7%
shoes-Sport 26.9%
BodyNormal 74.2%
shoes-White 22.7%
ub-Shirt 22.1%
ub-TShirt 22.7%
up-Gray 22.3%
hs-LongHair 19.1%
occlusionDown 19.4%
up-White 18.8%
faceLeft 17.2%
faceRight 17.6%
ub-Cotton 17.5%
up-Mixture 16.9%
shoes-Casual 15.6%
BodyFat 14.4%
lb-TightTrousers 13.4%
up-Blue 13.7%
action-CarrybyHand 12.8%
occlusion-Environment 12.5%
shoes-Boots 12.5%
ub-Sweater 12.5%
occlusion-Other 11.6%
shoes-Gray 10.4%
up-Red 10.8%
action-Gathering 9.2%
BodyThin 9.9%
low-Gray 9.0%
hs-Glasses 7.4%
attach-SingleShoulderBag 6.3%
lb-Dress 6.5%
lb-ShortSkirt 6.5%
lb-Skirt 6.5%
shoes-Brown 6.2%
Customer 94.5%
hs-BlackHair 94.5%
occlusionUp 6.0%
ub-Vest 5.8%
attach-Box 4.2%
Clerk 4.9%
occlusionLeft 4.1%
shoes-Red 4.3%
up-Green 4.7%
up-Yellow 4.3%
action-Calling 3.4%
action-Talking 3.3%
action-CarrybyArm 2.4%
action-Holding 2.5%
attach-HandBag 2.9%
attach-HandTrunk 2.4%
attach-PlasticBag 2.8%
low-Mixture 2.6%
low-Red 2.5%
low-White 2.6%
occlusion-Person 2.7%
occlusionRight 2.9%
shoes-Blue 3.0%
shoes-Mixture 2.4%
shoes-Yellow 2.7%
ub-SuitUp 2.2%
ub-Tight 3.0%
up-Brown 2.9%
up-Pink 2.7%
action-Pulling 1.7%
attach-Backpack 1.9%
attach-PaperBag 1.2%
hs-Hat 1.6%
hs-Muffler 1.1%
low-Green 1.6%
low-Yellow 1.9%
occlusion-Attachment 1.6%
shoes-Cloth 1.2%
shoes-Green 1.2%
ub-ShortSleeve 1.7%
up-Orange 1.1%
up-Purple 1.7%
action-Pusing 1.0%
AgeLess16 1.0%
hs-BaldHead 0.4%"



TMPATTRIBTRAINERR="/tmp/plot_elo_attr_train_err.tmp"
TMPATTRIBVALIDERR="/tmp/plot_elo_attr_valid_err.tmp"
TMPCLASSACCURACY="/tmp/plot_elo_class_accuracy.tmp"
OUTPLOTDIR="$LOGF.plot"

grep ^ATTRIBTRAINERROR "$LOGF" | sed 's/^ATTRIBTRAINERROR//' > "$TMPATTRIBTRAINERR"
grep ^ATTRIBVALIDERROR "$LOGF" | sed 's/^ATTRIBVALIDERROR//' > "$TMPATTRIBVALIDERR"
grep ^ATTRIBCLASSACCURACY "$LOGF" | sed 's/^ATTRIBCLASSACCURACY//' > "$TMPCLASSACCURACY"

mkdir "$OUTPLOTDIR"

HEADER="$(head -1 "$TMPATTRIBTRAINERR" | sed 's/  */ /g' | sed 's/^ //')"
ATTRIBUTES="$(echo "$HEADER" | (read first allother ; echo $allother))"

COLNBR=2
for ATTRNAME in $ATTRIBUTES
do
  echo "drawing attribute '$ATTRNAME'"

  OUTPLOTFILE="$OUTPLOTDIR/$ATTRNAME.png"

  ATTRFREQ="$(echo "$ATTR_TRAIN_FREQ" | grep "$ATTRNAME" | cut -d\  -f2)"

  echo "
  set terminal png size 640,480 enhanced font \"Helvetica,16\"
  set output \"$OUTPLOTFILE\"
  set title \"${ATTRNAME}   ${ATTRFREQ}\"
  set style line 1 lt 1 lw 2 pt 1 ps 0.5;
  set style line 2 lt 3 lw 2 pt 1 ps 0.5;
  set style line 3 lt 7 lw 1 pt 1 ps 0.5;
  set style line 4 lt 4 lw 2 pt 4 ps 0.5;
  set style line 5 lt 5 lw 2 pt 4 ps 0.5;
  set style line 6 lt 2 lw 2 pt 2 ps 0.5;
  set style line 7 lt 6 lw 2 pt 4 ps 0.5;
  set style line 8 lt 8 lw 2 pt 4 ps 0.5;
  set style line 20 lt 7 lw 2 pt 4 ps 0.5;
  set style line 100 lt 1 lc rgb \"gray\" lw 1
  set xlabel \"Batch number\"
  set ylabel \"Classif-error\"
  set yrange [0:100]
  set ytics 10
  set mytics 2
  set grid mytics ytics ls 100, ls 100
  #set autoscale y
  plot \"$TMPATTRIBTRAINERR\" using 1:$COLNBR title \"Train error\" with lines ls 2, \
       \"$TMPATTRIBVALIDERR\" using 1:$COLNBR title \"Valid error\" with lines ls 1, \
       \"$TMPCLASSACCURACY\"  using 1:(100 * \$$COLNBR) title \"Class accuracy\" with lines ls 3
  " | gnuplot

  COLNBR="$(expr 1 + $COLNBR)"
done

