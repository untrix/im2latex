#!/bin/bash
LOGFILE=~/logs/tensorboard2.out
cd ~
nohup tensorboard --logdir ~/im2latex/src/tb_metrics_dev --purge_orphaned_data --port 50003 > $LOGFILE 2>&1 &
tail -f $LOGFILE
