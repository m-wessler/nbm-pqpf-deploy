import re, os
import requests

import numpy as np
import pandas as pd
from datetime import datetime

from glob import glob
from functools import partial
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

base_url = 'https://noaa-nbm-grib2-pds.s3.amazonaws.com'

fhrStack = np.concatenate(
    [np.arange(3, 24, 3),
     np.arange(24, 72, 6),
     np.arange(72, 168.1, 12)]).astype(np.int)

def subset_grib_spatial(full_file):

    subset_file = full_file.replace('.co.grib2', '.WR.grib2')

    # Generate subset file using wgrib2
    nlon, xlon, nlat, xlat = -125, -100, 30, 50

    wgrib2 = '/usr/local/sbin/wgrib2'

    run_cmd = '%s %s -small_grib %d:%d %d:%d %s > ./tmp.txt'%(
        wgrib2, full_file, nlon, xlon, nlat, xlat, subset_file)

    os.system(run_cmd)

    if os.path.isfile(subset_file):
        os.remove(full_file)

def download_grib_subset(init,
    searchString='GUST', outDir='/data/nbmWinds/grib/', verbose=False):

    if not os.path.isdir(outDir):
        os.makedirs(outDir, exist_ok=True)

    init = pd.to_datetime(init)
    ymd = init.strftime('%Y%m%d')

    for fhr in fhrStack:

        gribURL = (f'{base_url}/blend.{ymd}/{init.hour:02d}/core/' +
            f'blend.t{init.hour:02d}z.core.f{fhr:03d}.co.grib2')

        # What should we name the file we save this data to?
        runDate = ymd
        outFile = '_'.join([runDate, searchString, gribURL.split('/')[-1]])
        outFile = outDir + outFile

        if not os.path.isfile(outDir + outFile):

            r = requests.get(gribURL + '.idx')
            lines = r.text.split('\n')

            # Search expression
            expr = re.compile(searchString)

            # Store the byte ranges in a dictionary
            byte_ranges = {}
            for n, line in enumerate(lines, start=1):

                # n is the line number (starting from 1) so that when we call for
                # `lines[n]` it will give us the next line. (Clear as mud??)

                # Use the compiled regular expression to search the line
                if expr.search(line):
                    # aka, if the line contains the string we are looking for...

                    # Get the beginning byte in the line we found
                    parts = line.split(':')
                    rangestart = int(parts[1])

                    # Get the beginning byte in the next line...
                    if n+1 < len(lines):
                        # ...if there is a next line
                        parts = lines[n].split(':')
                        rangeend = int(parts[1])
                    else:
                        # ...if there isn't a next line, then go to the end of the file.
                        rangeend = ''

                    # Store the byte-range string in our dictionary,
                    # and keep the line information too so we can refer back to it.

                    if 'std dev' not in line:
                        byte_ranges[f'{rangestart}-{rangeend}'] = line

            for i, (byteRange, line) in enumerate(byte_ranges.items()):

                if i == 0:
                    # If we are working on the first item, overwrite the existing file.
                    curl = f'curl -s --range {byteRange} {gribURL} > {outFile}'
                else:
                    # If we are working on not the first item, append the existing file.
                    curl = f'curl -s --range {byteRange} {gribURL} >> {outFile}'

                num, byte, date, var, level, forecast, *_ = line.split(':')

                if verbose:
                    print(f'  Downloading GRIB line [{num:>3}]: variable={var}' +
                          f', level={level}, forecast={forecast}')
                os.system(curl)

                subset_grib_spatial(outFile)
        else:
            subset_grib_spatial(outFile)

    print(f'{init} complete...')
    return None

start = datetime(2020, 10, 1, 0, 0)
end = datetime(2021, 9, 16, 0, 0)

downloadDates = pd.date_range(start, end, freq='6H')
print(downloadDates)

with ThreadPool(24) as tp:
    tp.map(download_grib_subset, downloadDates)
    tp.close()
    tp.join()

flist_raw = np.array(sorted(glob('./NBM_wind_grids/*')))

flist = [(f.split('/')[-1].split('_')[0],
  f.split('/')[-1].split('_')[-1].split('.')[1][1:3],
  f.split('/')[-1].split('_')[-1].split('.')[-3][1:])
  for f in flist_raw]

dlist = np.array(sorted(np.unique(
    [datetime.strptime(''.join(d[:2]),'%Y%m%d%H') for d in flist])))

missings_inits = [d for d in dlist.astype('datetime64[ns]') if d not in downloadDates]
print('Missing:\n', missings_inits)
