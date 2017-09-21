#!/usr/bin/env python2

import dl_commons as dlc
import tf_commons as tfc
import hyper_params as hp
import pprint
import numpy as np

hyper = hp.make_hyper({'build_image_context':False})
# print hyper.pformat()
t = hyper.to_table()
print t.dtype, t.shape
print t
#pprint.pprint(hyper.flatten())
