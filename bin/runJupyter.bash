#!/bin/bash
LOGFILE=~/logs/jupyter.out
cd ~
nohup jupyter notebook --no-browser > $LOGFILE 2>&1 &
tail -f $LOGFILE
