#!/usr/bin/env python

import time
from datetime import datetime
import pytz
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

from astropy.modeling.models import Gaussian2D, Polynomial2D, Moffat2D
from astropy.modeling.fitting import LevMarLSQFitter

from mmtwfs.wfs import wfsfind
from mmtwfs.custom_exceptions import WFSAnalysisFailed


tz = pytz.timezone("America/Phoenix")


def check_image(f, wfskey=None):
    hdr = {}
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        with fits.open(f) as hdulist:
            for h in hdulist:
                hdr.update(h.header)
            data = hdulist[-1].data
        if len(warns) > 0:
            for warn in warns:
                print(f"Got warning when opening {pathstr}: {str(warn.message)}")

    # if wfskey is None, figure out which WFS from the header info...
    if wfskey is None:
        # check for MMIRS
        if 'WFSNAME' in hdr:
            if 'mmirs' in hdr['WFSNAME']:
                wfskey = 'mmirs'
        if 'mmirs' in f.name:
            wfskey = 'mmirs'

        # check for binospec
        if 'bino' in f.name:
            wfskey = 'binospec'
        if 'ORIGIN' in hdr:
            if 'Binospec' in hdr['ORIGIN']:
                wfskey = 'binospec'

        # check for new F/9
        if 'f9wfs' in f.name:
            wfskey = 'newf9'
        if 'OBSERVER' in hdr:
            if 'F/9 WFS' in hdr['OBSERVER']:
                wfskey = 'newf9'
        if wfskey is None and 'CAMERA' in hdr:
            if 'F/9 WFS' in hdr['CAMERA']:
                wfskey = 'newf9'

        # check for old F/9
        if 'INSTRUME' in hdr:
            if 'Apogee' in hdr['INSTRUME']:
                wfskey = 'oldf9'
        if 'DETECTOR' in hdr:
            if 'Apogee' in hdr['DETECTOR']:
                wfskey = 'oldf9'

        # check for F/5 (hecto)
        if wfskey is None and 'SEC' in hdr:  # mmirs has SEC in header as well and is caught above
            if 'F5' in hdr['SEC']:
                wfskey = 'f5'
        if Path(f.parent / "F5").exists():
            wfskey = 'f5'

    if wfskey is None:
        # if wfskey is still None at this point, whinge.
        print(f"Can't determine WFS for {f.name}...")

    if 'AIRMASS' not in hdr:
        if 'SECZ' in hdr:
            hdr['AIRMASS'] = hdr['SECZ']
        else:
            hdr['AIRMASS'] = np.nan

    # we need to fix the headers in all cases to have a proper DATE-OBS entry with
    # properly formatted FITS timestamp.  in the meantime, this hack gets us what we need
    # for analysis in pandas.
    dtime = None
    if 'DATEOBS' in hdr:
        dateobs = hdr['DATEOBS']
        if 'UT' in hdr:
            ut = hdr['UT']
        elif 'TIME-OBS' in hdr:
            ut = hdr['TIME-OBS']
        else:
            ut = "07:00:00"  # midnight
        timestring = dateobs + " " + ut + " UTC"
        if wfskey in ('newf9', 'f5'):
            if '-' in timestring:
                dtime = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S %Z")
            else:
                dtime = datetime.strptime(timestring, "%a %b %d %Y %H:%M:%S %Z")

        else:
            dtime = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S %Z")
    else:
        if wfskey == "oldf9":
            d = hdr['DATE-OBS']
            if '/' in d:
                day, month, year = d.split('/')
                year = str(int(year) + 1900)
                timestring = year + "-" + month + "-" + day + " " + hdr['TIME-OBS'] + " UTC"
            else:
                timestring = d + " " + hdr['TIME-OBS'] + " UTC"
            dtime = datetime.strptime(timestring, "%Y-%m-%d %H:%M:%S %Z")
        else:
            if 'DATE-OBS' in hdr:
                timestring = hdr['DATE-OBS'] + " UTC"
                try:
                    dtime = datetime.strptime(timestring, "%Y-%m-%dT%H:%M:%S.%f %Z")
                except:
                    dtime = datetime.strptime(timestring, "%Y-%m-%dT%H:%M:%S %Z")
            else:
                dt = datetime.fromtimestamp(f.stat().st_ctime)
                local_dt = tz.localize(dt)
                dtime = local_dt.astimezone(pytz.utc)

    if dtime is None:
        print(f"No valid timestamp in header for {f.name}...")
        obstime = None
    else:
        obstime = dtime.isoformat().replace('+00:00', '')

    hdr['WFSKEY'] = wfskey
    hdr['OBS-TIME'] = obstime
    return data, hdr


findpars = {
    "oldf9": {"fwhm": 8.0, "thresh": 5.0},
    "newf9": {"fwhm": 12.0, "thresh": 7.0},
    "f5": {"fwhm": 9.0, "thresh": 5.0},
    "mmirs": {"fwhm": 7.0, "thresh": 5.0},
    "binospec": {"fwhm": 7.0, "thresh": 5.0}
}


def process_image(f, clobber=False):
    tablefile = Path(str(f.parent / f.stem) + ".csv")
    pathstr = str(Path(str(f.parent / f.stem)))

    if '_ave' in str(f):
        print(f"Not processing {f.name} because it's an average of multiple images")
        return

    if tablefile.exists() and not clobber:
        print(f"{f.name} already processed...")
        return
    else:
        print(f"Processing {f.name}...")

    try:
        data, hdr = check_image(f)
        w = hdr['WFSKEY']
        mean, median, stddev = stats.sigma_clipped_stats(data, sigma=3.0, iters=None)
        data = data - median
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            spots, fig = wfsfind(data, fwhm=findpars[w]['fwhm'], threshold=findpars[w]['thresh'], plot=False)
            if len(warns) > 0:
                for warn in warns:
                    print(f"Got warning when finding spots in {pathstr}: {str(warn.message)}")
    except WFSAnalysisFailed as e:
        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                spots, fig = wfsfind(data, fwhm=2.*findpars[w]['fwhm'], threshold=findpars[w]['thresh'], plot=False)
                if len(warns) > 0:
                    for warn in warns:
                        print(f"Got warning when finding spots with 2x larger FWHM in {pathstr}: {str(warn.message)}")
        except Exception as e:
            print(f"Failed to find spots for {pathstr}: {e}")
            return
    except Exception as ee:
        print(f"Failure in data loading or spot finding in {pathstr}: {ee}")
        return

    apsize = 15.
    apers = photutils.CircularAperture(
            (spots['xcentroid'], spots['ycentroid']),
            r=apsize
    )
    masks = apers.to_mask(method="subpixel")
    props = []
    spot_lines = []
    fit_lines = []
    for m, s in zip(masks, spots):
        tline = {}
        subim = m.cutout(data)
        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                props_table = photutils.data_properties(subim).to_table()
                if len(warns) > 0:
                    for warn in warns:
                        print(f"Got warning when measuring spots with photutils in {pathstr}: {str(warn.message)}")
        except Exception as e:
            print(f"Can't measure source properties for {pathstr}: {e}")
            continue
        moment_fwhm = 0.5 * (props_table['semimajor_axis_sigma'][0].value + props_table['semiminor_axis_sigma'][0].value) * stats.gaussian_sigma_to_fwhm
        props.append(props_table)
        spot_lines.append(s)
        tline['filename'] = f.name
        for k in hdr:
            if 'COMMENT' not in k and k != '':
                tline[k] = hdr[k]

        y, x = np.mgrid[:subim.shape[0], :subim.shape[1]]
        sigma = (props_table['semimajor_axis_sigma'][0].value + props_table['semiminor_axis_sigma'][0].value) / 2.

        fitter = LevMarLSQFitter()
        gauss_model = Gaussian2D(
            amplitude=subim.max(),
            x_mean=subim.shape[1]/2.,
            y_mean=subim.shape[0]/2.,
            x_stddev = sigma,
            y_stddev = sigma
        ) + Polynomial2D(degree=0)
        moffat_model = Moffat2D(
            amplitude=subim.max(),
            x_0=subim.shape[1]/2.,
            y_0=subim.shape[0]/2.,
            gamma=sigma
        ) + Polynomial2D(degree=0)

        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                gauss_fit = fitter(gauss_model, x, y, subim)
                if len(warns) > 0:
                    for warn in warns:
                        print(f"Got warning when fitting 2D gaussian to spots in {pathstr}: {str(warn.message)}")
            gauss_resid = subim - gauss_fit(x, y)
            gauss_fwhm = 0.5 * (gauss_fit.x_stddev_0.value + gauss_fit.y_stddev_0.value) * stats.gaussian_sigma_to_fwhm
            tline['gauss_x'] = gauss_fit.x_mean_0.value
            tline['gauss_y'] = gauss_fit.y_mean_0.value
            tline['gauss_sigx'] = gauss_fit.x_stddev_0.value
            tline['gauss_sigy'] = gauss_fit.y_stddev_0.value
            tline['gauss_amplitude'] = gauss_fit.amplitude_0.value
            tline['gauss_theta'] = gauss_fit.theta_0.value
            tline['gauss_background'] = gauss_fit.c0_0_1.value
            tline['gauss_rms'] = np.nanstd(gauss_resid)
            tline['gauss_fwhm'] = gauss_fwhm
        except Exception as e:
            print(f"Gaussian fit failed in {pathstr}: {e}")
            tline['gauss_x'] = np.nan
            tline['gauss_y'] = np.nan
            tline['gauss_sigx'] = np.nan
            tline['gauss_sigy'] = np.nan
            tline['gauss_amplitude'] = np.nan
            tline['gauss_theta'] = np.nan
            tline['gauss_background'] = np.nan
            tline['gauss_rms'] = np.nan
            tline['gauss_fwhm'] = np.nan

        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                moffat_fit = fitter(moffat_model, x, y, subim)
                if len(warns) > 0:
                    for warn in warns:
                        print(f"Got warning when fitting 2D Moffat to spots in {pathstr}: {str(warn.message)}")
            moffat_resid = subim - moffat_fit(x, y)
            gamma = moffat_fit.gamma_0.value
            alpha = moffat_fit.alpha_0.value
            moffat_fwhm = np.abs(2. * gamma * np.sqrt(2.**(1./alpha) - 1.))
            tline['moffat_amplitude'] = moffat_fit.amplitude_0.value
            tline['moffat_gamma'] = gamma
            tline['moffat_alpha'] = alpha
            tline['moffat_x'] = moffat_fit.x_0_0.value
            tline['moffat_y'] = moffat_fit.y_0_0.value
            tline['moffat_background'] = moffat_fit.c0_0_1.value
            tline['moffat_rms'] = np.nanstd(moffat_resid)
            tline['moment_fwhm'] = moment_fwhm
            tline['moffat_fwhm'] = moffat_fwhm
        except Exception as e:
            print(f"Moffat fit failed in {pathstr}: {e}")
            tline['moffat_amplitude'] = np.nan
            tline['moffat_gamma'] = np.nan
            tline['moffat_alpha'] = np.nan
            tline['moffat_x'] = np.nan
            tline['moffat_y'] = np.nan
            tline['moffat_background'] = np.nan
            tline['moffat_rms'] = np.nan
            tline['moment_fwhm'] = np.nan
            tline['moffat_fwhm'] = np.nan
        fit_lines.append(tline)

    fit_table = Table(fit_lines)
    spot_table = Table(vstack(spot_lines))
    prop_table = Table(vstack(props))
    t = hstack([spot_table, prop_table, fit_table])
    try:
        if tablefile.exists():
            if clobber:
                t.write(tablefile, format="csv")
        else:
            t.write(tablefile, format="csv")
    except Exception as e:
        print(f"Failed to write {str(tablefile)}: {e}")
    return t


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dirs', help="Glob of directories to look for WFS data.")
parser.add_argument('-r', '--root', default="/Volumes/LACIE SHARE/wfsdat", help="Root directory for WFS data.")
args = parser.parse_args()

rootdir = Path(args.root)
dirs = sorted(list(rootdir.glob(args.dirs)))

nproc = 8
for d in dirs:
    if d.is_dir():
        print(f"Working in {d.name}...")
        fitsfiles = sorted(list(d.glob("*.fits")))
        with Pool(processes=nproc) as pool:  # my mac's i7 has 4 cores + hyperthreading so 8 virtual cores.
            pool.map(process_image, fitsfiles)  # plines comes out in same order as fitslines!
