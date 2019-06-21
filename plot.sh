#!/bin/ksh

if [ $# -ne 1 ]
then
	echo "usage: $0 <jobnr>"
	exit 1
fi

LOGF=$(ls ../../LOGS/logfile_jobid_"$1"*.txt)

if [ ! -f "$LOGF" ]
then
	echo "File does not exit: $LOGF"
	exit 1
fi

cat $LOGF  | grep ^Epoch | awk ' { print $6 }' > /tmp/train_loss
cat $LOGF  | grep ^Epoch | awk ' { print $8 }' > /tmp/train_err
cat $LOGF  | grep ^Epoch | awk ' { print $10 }' > /tmp/test_err


echo '
set style line 1 lt 1 lw 2 pt 1 ps 0.5;
set style line 2 lt 3 lw 2 pt 1 ps 0.5;
set style line 3 lt 7 lw 1 pt 1 ps 0.5;
set style line 4 lt 4 lw 2 pt 4 ps 0.5;
set style line 5 lt 5 lw 2 pt 4 ps 0.5;
set style line 6 lt 2 lw 2 pt 2 ps 0.5;
set style line 7 lt 6 lw 2 pt 4 ps 0.5;
set style line 8 lt 8 lw 2 pt 4 ps 0.5;
set style line 20 lt 7 lw 2 pt 4 ps 0.5;
set ylabel "Classif-error"
set yrange [0:110]
set y2label "Loss"
set autoscale y2
set y2tics 
set logscale y2
plot "/tmp/train_err" title "Train error (on batch)" with lines ls 1, "/tmp/test_err" title "Valid error" with lines ls 2, "/tmp/train_loss" title "Train loss (logscale)" with lines ls 3 axes x1y2
' | gnuplot -persist


export LANG=C
echo -n "Min train loss:"
cat /tmp/train_loss | awk ' BEGIN {min=999999;} { if ($1<min) min=$1; } END { print min }'
echo -n "Min train error par batch:"
cat /tmp/train_err | awk ' BEGIN {min=999999;} { if ($1<min) min=$1; } END { print min }'
echo -n "Min test error:"
cat /tmp/test_err | awk ' BEGIN {min=999999;} { if ($1<min) min=$1; } END { print min }'

echo "---"

echo -n "Average of last 10 train losses per batch:"
cat /tmp/train_loss | tail -10 | awk ' BEGIN {sum=0;} { sum=sum+$1; } END { print sum/10.0 }'
echo -n "Average of last 10 train errors per batch:"
cat /tmp/train_err | tail -10 | awk ' BEGIN {sum=0;} { sum=sum+$1; } END { print sum/10.0 }'
echo -n "Average of last 10 test errors:"
cat /tmp/test_err | tail -10 | awk ' BEGIN {sum=0;} { sum=sum+$1; } END { print sum/10.0 }'
