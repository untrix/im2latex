#!/bin/bash
LOGFILE=~/logs/run.out
cd ~
rm LOGFILE
nohup ./run.py -e -1 -b 64 -w 10 -k 1.0 -p -i 0 >$LOGFILE  2>&1 &
tail -f $LOGFILE
