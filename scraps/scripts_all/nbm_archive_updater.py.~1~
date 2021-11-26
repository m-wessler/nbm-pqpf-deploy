import shutil
import shlex
import subprocess
import sys, os, gc
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing as mp

from glob import glob
from io import StringIO
from datetime import datetime, timedelta

core_limit = 8

nbm_dir = '/scratch/general/lustre/u1070830/nbm/'
# urma_dir = '/scratch/general/lustre/u1070830/urma/'

def unpack_fhr(nbm_file):
    import pygrib
    
    nlat, xlat = 30, 50
    nlon, xlon = -130, -100

    if os.path.isfile(nbm_file):

        with pygrib.open(nbm_file) as grb:

            try:
                lats, lons = grb.message(1).latlons()
            except:
                data = None
            else:
                idx = np.where(
                    (lats >= nlat) & (lats <= xlat) &
                    (lons >= nlon) & (lons <= xlon))

                init_time = nbm_file.split('/')[-2:]
                init_time = init_time[0] + init_time[1].split('.')[1][1:3]
                init_time = datetime.strptime(init_time, '%Y%m%d%H')
                valid_fhr = int(os.path.basename(nbm_file).split('/')[-1].split('.')[3][1:])

                # Check if nbm3.2
                if init_time.hour in [1, 7, 13, 19]:
                    init_time -= timedelta(hours=1)
                    valid_fhr += 1

                valid_time = init_time + timedelta(hours=valid_fhr)
                #print(init_time, valid_fhr, valid_time)
                #print('\t', valid_fhr, valid_time)

                percentile, probability, deterministic = [], [], []
                percentile_labels, probability_labels, deterministic_labels = [], [], []

                data = []
                for msg in grb.read():

                    interval = msg['stepRange'].split('-')
                    interval = int(interval[1]) - int(interval[0])

                    if interval == 24:

                        if 'Probability of event' in str(msg):

                            threshold = round(msg['upperLimit']/25.4, 2)

                            if threshold in [0.01, 0.10, 0.25, 0.50, 1.00, 2.00]:

                                idata = xr.DataArray(msg.data()[0].astype(np.float32), name='probx',
                                                     dims=('y', 'x'), 
                                                     coords={'lat':(('y', 'x'), lats), 
                                                             'lon':(('y', 'x'), lons)})

                                idata['init'] = init_time
                                idata['valid'] = valid_time
                                idata['fhr'] = valid_fhr
                                idata['interval'] = interval
                                idata['threshold'] = threshold

                                data.append(idata)

        gc.collect()

        try:
            data = xr.concat(data, dim='threshold')

        except:
            return None

        else:
            data_slice = data.isel(x=slice(idx[1].min(), idx[1].max()), 
                  y=slice(idx[0].min(), idx[0].max()))

            return data_slice

    else:
        return None
    
def combine_nbm_old_new(fhr): 

    fhr_agg_old_file = sorted(glob(nbm_dir + 'extract/' + '*fhr%03d.nc'%fhr))[0]
    fhr_agg_new_file = sorted(glob(nbm_dir + 'extract_new/' + '*fhr%03d.new.nc'%fhr))[0]
    
    fhr_agg_old = xr.open_dataset(fhr_agg_old_file)
    fhr_agg_new = xr.open_dataset(fhr_agg_new_file)

    not_duplicated = np.array([t for t in fhr_agg_new.valid.values if t not in fhr_agg_old.valid.values])
    fhr_agg_new = fhr_agg_new.sel(valid=not_duplicated)
    
    print('Combining new FHR%03d data...'%fhr)
    
    fhr_agg = xr.concat([fhr_agg_old, fhr_agg_new], dim='valid')
    
    fhr_agg_output_file = nbm_dir + 'extract_new/' + os.path.basename(fhr_agg_old_file)
    fhr_agg.to_netcdf(fhr_agg_output_file)
    print('Saved: %s'%fhr_agg_output_file)
    
    return None
    
if __name__ == '__main__':

    # urma_raw = np.array(sorted(glob(urma_dir + '*.grib2')))
    # urma_agg = xr.open_dataset(glob(urma_dir + 'agg/*')[0])

    nbm_raw = np.array(sorted([f for f in sorted(glob(nbm_dir + '*/*.grib2')) if 'extract' not in f]))
    nbm_agg = np.array(sorted(glob(nbm_dir + 'extract/*')))

    # # Last complete URMA aggregated
    # last_urma_agg = urma_agg.valid[-1].values
    # last_urma_agg

    # # Last complete URMA downloaded
    # last_urma_download = xr.open_dataset(urma_raw[-1], engine='cfgrib').valid_time.values
    # last_urma_download

    # Last complete run aggregated
    nbm_agg_file = nbm_agg[0]
    last_nbm_agg = xr.open_dataset(nbm_agg_file).init[-1].values

    # Last complete run downloaded
    nbm_f024 = np.array(sorted([f for f in nbm_raw if 'f024' in f]))[-1]
    last_nbm_download = xr.open_dataset(nbm_f024, engine='cfgrib').valid_time.values

    # # Since we want this to update as soon as possible, use the 24h time mark
    # # But we can't produce anything unless URMA is updated as well
    # newest_time_match = np.min([last_nbm_download, last_urma_download])
    # newest_time_match

    # # Figure out how many missing inits between newest available data and last aggregate
    # newest_agg_time_match = np.min([last_nbm_agg, last_urma_agg])
    # newest_agg_time_match

    nbm_upload_lag = 6
    most_recent_time = datetime.utcnow() #+ timedelta(hours=4, minutes=10)

    roundUp = True if most_recent_time.minute < 30 else False

    most_recent_time = most_recent_time.replace(minute=0, second=0, microsecond=0)

    if roundUp:
        most_recent_time += timedelta(hours=1)

    # Round down to nearest 0, 6, 12, 18, then grab the run prior
    most_recent_time -= timedelta(hours=(most_recent_time.hour%6)+6)

    # if last_nbm_download > last_nbm_agg:
    if np.datetime64(most_recent_time) > last_nbm_agg:

        print('Newer NBM Available\n')    
        # fill_runs = pd.date_range(last_nbm_agg, last_nbm_download, freq='6H')
        fill_runs = pd.date_range(last_nbm_agg, most_recent_time, freq='6H')[1:]

    print('Runs to fill: %s'%fill_runs)

    # Call the NBM download here
    python = '/uufs/chpc.utah.edu/common/home/u1070830/anaconda3/envs/xlab/bin/python '
    dl_script = '/uufs/chpc.utah.edu/common/home/u1070830/code/model-tools/nbm/get_nbm_gribs_aws.py ' 

    dl_start, dl_end = pd.to_datetime(fill_runs[0]), pd.to_datetime(fill_runs[-1])
    dl_start, dl_end = [datetime.strftime(t, '%Y%m%d%H%M') for t in [dl_start, dl_end]]

    cmd = python + dl_script + '%s %s'%(dl_start, dl_end)
    
    subprocess.run(shlex.split(cmd), stderr=sys.stderr, stdout=sys.stdout)
    
    for forecast_hour in np.arange(24, 168+1, 24):

        outdir = nbm_dir + 'extract_new/'
        os.makedirs(outdir, exist_ok=True)
        outfile = 'nbm_probx_fhr%03d.new.nc'%forecast_hour

        if not os.path.isfile(outdir+outfile):

            flist = []
            for init in fill_runs:

                search_str = nbm_dir + '%s/*t%02dz*f%03d*WR.grib2'%(
                    init.strftime('%Y%m%d'), init.hour, forecast_hour)
                search = glob(search_str)

                if len(search) > 0:
                    flist.append(search[0])

            flist = np.array(sorted(flist))
            print('nfiles: ', len(flist))

            ncores = np.min([core_limit, len(flist)])
            with mp.get_context('fork').Pool(ncores) as p:
                returns = p.map(unpack_fhr, flist, chunksize=1)
                p.close()
                p.join()

            returns = [item for item in returns if item is not None]
            returns = xr.concat(returns, dim='valid')

            returns.to_netcdf(outdir + outfile)
            print('Saved %s'%(outdir + outfile))

            del returns
            gc.collect()
            
    nbm_agg_new = np.array(sorted(glob(nbm_dir + 'extract_new/*')))

    ncores = np.min([core_limit, len(np.arange(24, 168.1, 24))])
    with mp.get_context('fork').Pool(ncores) as p:
        returns = p.map(combine_nbm_old_new, np.arange(24, 168.1, 24), chunksize=1)
        p.close()
        p.join()
        
    # Verify that the new file didn't corrupt the existing data!
    new_agg_temp = sorted([f for f in glob(nbm_dir + 'extract_new/*.nc') if '.new.' in f])
    new_agg_check = sorted([f for f in glob(nbm_dir + 'extract_new/*.nc') if '.new.' not in f])
    old_agg_check = sorted(glob(nbm_dir + 'extract/*.nc'))

    for temp_file, new_file, old_file in zip(new_agg_temp, new_agg_check, old_agg_check):

        try:
            new = xr.open_dataset(new_file)
            old = xr.open_dataset(old_file)

        except:
            pass

        else:
            if new.valid[-1] > old.valid[-1]:
                print('New aggregate %s updated, check...'%os.path.basename(new_file))

                try:
                    os.remove(temp_file)
                    print(temp_file, '-->', 'deleted')
                except:
                    pass

                try:
                    shutil.move(old_file, old_file.replace('.nc', '.old.nc'))
                    print(old_file, '-->', old_file.replace('.nc', '.old.nc'))
                except:
                    pass

                try:
                    shutil.move(new_file, new_file.replace('extract_new', 'extract'))
                    print(new_file, '-->', new_file.replace('extract_new', 'extract'))
                except:
                    pass
                
                try:
                    os.remove(old_file.replace('.nc', '.old.nc'))
                    print(old_file.replace('.nc', '.old.nc'), '-->', 'deleted')
                except:
                    pass

            else:
                print('New aggregate %s failed, follow up...'%os.path.basename(new_file))

            print()
    
    print('\nDone...')