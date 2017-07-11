#!/bin/bash
pgrep -fl $1
pkill -fl $1
pgrep -fl $1