#!/bin/bash
LOGFILE=~/logs/jupyter_lab.out
cd ~
nohup jupyter lab --ip=* --port 50003 > $LOGFILE 2>&1 &
tail -f $LOGFILE
