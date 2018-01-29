#!/usr/bin/env python2
import os
import argparse as arg

parser = arg.ArgumentParser(description='Builds markdown content listing filenames passed in')
parser.add_argument("--filenames", "-f", dest="filenames", nargs="*",
                    help="Space separated list of filenames in local folder to include in the index")
parser.add_argument("-o", dest="outname", type=str,
                    help="Output file name, will be created in local folder")

args = parser.parse_args()
filenames = args.filenames
lines = []

for f in filenames:
    line = '1. [%s](%s)\n'%(f.strip('.html').strip('alpha_').strip('_gray'), f)
    lines.append(line)
    print(line)

with open(args.outname, 'w') as f:
    f.write('---\ntitle: Attentive Scan\n---\n### Examples of Attentive Scan\n')
    f.writelines(lines)

