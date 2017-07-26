#!/bin/bash
LOGFILE=~/logs/jupyter.out
cd ~
nohup jupyter notebook --no-browser --port 50001 > $LOGFILE 2>&1 &
tail -f $LOGFILE
