#!/bin/bash
LOGFILE=~/logs/run.out
cd ~
rm LOGFILE
##  ./run.py -e -1 -b 40 -p -i 0 --r-lambda 0.00005 -k 1.0 -w 10 --squash-input-seq  --logdir ./tb_metrics_dev
## ./run.py -e -1 -b 40 -p -i 0 --r-lambda 0.00005 -k 1.0 -w 10 --squash-input-seq --logdir ./tb_metrics_dev --logdir-tag test_3.1LSTM_2init_3out_3attConv_1beta
nohup ./run.py -e -1 -b 64 -w 10 -k 1.0 -p -i 0 >$LOGFILE  2>&1 &
tail -f $LOGFILE
