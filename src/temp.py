#!/usr/bin/env python2

import os
import pandas as pd
import numpy as np
import data_commons as dtc
import dl_commons as dlc
import hyper_params as hp
import viz_commons as vc
from viz_commons import VisualizeDir, DiffParams, VisualizeStep

vd = VisualizeDir('/zpool_3TB/i2l/tb_metrics/2017-10-20 23-45-40 PDT reset_3.1LSTM_2init_3out_3attConv_1beta/snapshot-00078736/store_2')
vs = VisualizeStep(vd, 'validation', 78736)

strs = vs.strs('predicted_ids', trim=True)
