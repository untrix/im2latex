#!/bin/bash
LOGFILE=~/logs/spyder.out
cd ~
nohup spyder > $LOGFILE 2>&1 &
tail -f $LOGFILE
