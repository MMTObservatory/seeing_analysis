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
import pandas as pd

import photutils
import astropy.units as u
from astropy import stats
from astropy.io import fits
from astropy.table import Table, hstack, vstack


outputs = ['spot_reduce_mean.csv', 'spot_reduce_std.csv', 'spot_reduce_median.csv']

def process_directory(directory):
    print(f"Processing {str(directory)}...")
    csvs = sorted(list(directory.glob("*.csv")))
    means = []
    stds = []
    medians = []
    for c in csvs:
        if "spot_reduce" not in str(c):
            df = pd.read_csv(c)
            if len(df) > 0:
                mean = df.mean()
                std = df.std()
                median = df.median()
                for d in [mean, std, median]:
                    d['obstime'] = df['OBS-TIME'][0]
                    d['filename'] = df['filename'][0]
                means.append(mean)
                stds.append(std)
                medians.append(median)
    mean_df = pd.concat(means, axis=1, sort=True)
    std_df = pd.concat(stds, axis=1, sort=True)
    median_df = pd.concat(medians, axis=1, sort=True)
    for df, f in zip([mean_df, std_df, median_df], outputs):
        td = df.T
        td = td.set_index('filename')
        td.to_csv(directory / f, index=True)

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
