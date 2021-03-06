#!/usr/bin/env python

import time
from datetime import datetime
import os
import sys
import argparse
import warnings

import multiprocessing
from multiprocessing import Pool

from pathlib import Path

import numpy as np

import photutils
import astropy.units as u
from astropy import stats
from astropy.io import fits
from astropy.table import Table, hstack, vstack


def process_directory(directory):
    print(f"Processing {str(directory)}...")
    csvs = sorted(list(directory.glob("*.csv")))
    wfses = []
    for c in csvs:
        if "spot_reduce" not in str(c) and 'wfskeys' not in str(c):
            df = Table.read(c)
            if len(df) > 0:
                wfs = {"filename": df['filename'][0], "wfskey": df['WFSKEY'][0]}
                wfses.append(wfs)
    outdf = Table(wfses)
    outdf.write(directory / "wfskeys.csv", overwrite=True)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default="/Volumes/Seagate2TB/spot_analysis", help="Root directory for WFS spot data.")
parser.add_argument('-d', '--dirs', default="20*", help="Glob of directories to process.")
parser.add_argument('-n', '--nproc', default=8, type=int, help="Number of pool workers to spawn.")
args = parser.parse_args()

rootdir = Path(args.root)
dirs = sorted(list(rootdir.glob(args.dirs)))

nproc = args.nproc
with Pool(processes=nproc) as pool:  # my mac's i7 has 4 cores + hyperthreading so 8 virtual cores.
    pool.map(process_directory, dirs)  # plines comes out in same order as fitslines!

#for d in dirs:
#    process_directory(d)
