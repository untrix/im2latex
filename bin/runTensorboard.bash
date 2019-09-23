#!/bin/bash
LOGFILE=~/logs/tensorboard.out
cd ~
nohup tensorboard --logdir ~/predictions/logdir --purge_orphaned_data --port 50002 > $LOGFILE 2>&1 &
tail -f $LOGFILE
