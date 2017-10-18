#!/bin/bash
LOGFILE=~/logs/tensorboard.out
cd ~
nohup tensorboard --logdir ~/im2latex/src/tb_metrics_view --purge_orphaned_data --port 50002 > $LOGFILE 2>&1 &
tail -f $LOGFILE
