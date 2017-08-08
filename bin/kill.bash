#!/bin/bash
pgrep -fl $1
pkill -fa $1
pgrep -fl $1
