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

core_limit = 24

nbm_dir = '/nas/stid/data/nbm/pqpf/'

python = '/home/michael.wessler/anaconda3/envs/pqpf/bin/python '
dl_script = '/home/michael.wessler/code/get_nbm_gribs_aws.py ' 

def extract_fhr_data(f, interval=24, keep_percentiles=[]): 
    import pygrib

    data = []
    
    with pygrib.open(f) as grb:
                
        fhr = int(f.split('.')[3].replace('f', ''))
        
        for msg in grb.read():
            
            step = timedelta(hours=msg.endStep - msg.startStep)
            lead = msg.endStep
            
            if (fhr == lead) & (step == timedelta(hours=interval)):
                
                lats, lons = msg.latlons()

                if 'probability' in str(msg).lower():
                    
                    init = datetime.strptime(str(msg.dataDate) + '%04d'%msg.dataTime, '%Y%m%d%H%M')                    
                    valid = datetime.strptime(str(msg.validityDate) + '%04d'%msg.validityTime, '%Y%m%d%H%M')

                    threshold = msg.upperLimit
                    threshold_in = round(threshold/25.4, 2)

                    if threshold_in <= 4.0:
                        
                        idata = xr.DataArray([msg.data()[0].astype(np.float32)], name='probx',
                                                     dims=('valid', 'y', 'x'), 
                                                     coords={'valid':[valid],
                                                             'lat':(('y', 'x'), lats), 
                                                             'lon':(('y', 'x'), lons)})
                        idata['init'] = init                        
                        idata['interval'] = interval
                        idata['step'] = step
                        idata['fhr'] = lead
                        
                        idata['threshold'] = threshold
                        idata['threshold_in'] = threshold_in
                        
                        # Manually fix encoding issues...
                        idata['init'].encoding['units'] = 'hours since 2000-01-01 00:00:00'
                        idata['valid'].encoding['units'] = 'hours since 2000-01-01 00:00:00'
                        
                        data.append(idata)

                elif 'percentileValue' in msg.keys():
                    
                    if msg.percentileValue in keep_percentiles:
                        
                        # Append this data later with similar code as above, for now pass
                        # print(msg.percentileValue, msg)
                        
                        pass
    
    try:
        data = xr.concat(data, dim='threshold')    
    
    except:
        print(f, ' failed')
        #raise
    
    else:
        out_file = 'blend.%s.t%02dz.qmd.f%03d.WR.nc'%(init.strftime('%Y%m%d'), init.hour, lead)

        # Manually fix encoding issues...
        data['init'].encoding['units'] = 'hours since 2000-01-01 00:00:00'
        data['valid'].encoding['units'] = 'hours since 2000-01-01 00:00:00'
        
        data.to_netcdf(nbm_dir + 'extract/' + out_file, unlimited_dims='valid')
        print(out_file, ' saved')

        del data
        gc.collect()

    return None

if __name__ == '__main__':
    
    nbm_extract = np.array(sorted(glob(nbm_dir + 'extract/*.nc')))
    nbm_agg = np.array(sorted(glob(nbm_dir + 'agg/*.nc')))
    print(nbm_agg)

    # Most recent run in archive
    try:
        with xr.open_dataset(sorted([f for f in nbm_agg if 'f168' in f])[-1]) as sample:
            
            print(sample)

            if sample.init.size > 1:
                most_recent_agg = pd.to_datetime(sample.init[-1].values)
            else:
                most_recent_agg = pd.to_datetime(sample.init.values)
                
            most_recent_agg += timedelta(hours=12)
    except:
        raise
        # Start fresh/NBM 4.0 start date
        most_recent_agg = datetime(2020, 9, 30, 12, 0)
      
    # Start new archive  
    # most_recent_agg = datetime(2020, 9, 30, 12, 0)

    # Most recent run available
    nbm_upload_lag = 6
    most_recent_time = datetime.utcnow()

    roundUp = True if most_recent_time.minute < 30 else False

    most_recent_time = most_recent_time.replace(minute=0, second=0, microsecond=0)

    if roundUp:
        most_recent_time += timedelta(hours=1)

    # Round down to nearest 0, 6, 12, 18, then grab the run prior
    most_recent_time -= timedelta(hours=(most_recent_time.hour%6)+6)

    print('Fill from:', most_recent_agg)
    print('Fill to:', most_recent_time)
    # input('Hold... Press enter to continue...')

    # if last_nbm_download > last_nbm_agg:
    if np.datetime64(most_recent_time) > most_recent_agg:

        print('Newer NBM Available')    
        # fill_runs = pd.date_range(last_nbm_agg, last_nbm_download, freq='6H')
        fill_runs = pd.date_range(most_recent_agg, most_recent_time, freq='6H')[1:]

    print('\nFill runs:')
    print(fill_runs)

    dl_start, dl_end = pd.to_datetime(fill_runs[0]), pd.to_datetime(fill_runs[-1])
    dl_start, dl_end = [datetime.strftime(t, '%Y%m%d%H%M') for t in [dl_start, dl_end]]

    cmd = python + dl_script + '%s %s'%(dl_start, dl_end)
    print('\n%s\n'%cmd)
    subprocess.call(shlex.split(cmd))
    
    nbm_raw = np.array(sorted([f for f in sorted(glob(nbm_dir + '*/*.grib2'))]))

    for forecast_hour in np.arange(24, 168+1, 24):

        print('\nProcessing f%03d'%forecast_hour)

        list_new_extracts = []
        for run in fill_runs:        
                try:
                    list_new_extracts.append([f for f in nbm_raw if (
                        (run.strftime('%Y%m%d') in f) & 
                        ('t%02dz.qmd.f%03d'%(run.hour, forecast_hour) in f))][0])
                except:
                    pass

        list_new_extracts = sorted(list_new_extracts)
        
        workers = min(core_limit, len(list_new_extracts))
        print('Extracting %d NBM data files using %d processes'%(len(list_new_extracts), workers))

        # Singlethreaded for debugging
        # [extract_fhr_data(_f) for _f in list_new_extracts]

        with mp.get_context('fork').Pool(core_limit) as p:
            p.map(extract_fhr_data, list_new_extracts, chunksize=1)
            p.close()
            p.join()

        gc.collect()
        
        # Remove raw/temp NBM files
        # [os.remove(f) for f in list_new_extracts]

        list_new_extracts_nc = glob(nbm_dir + 'extract/*.nc')
        months = np.unique([f.split('.')[1][:6] for f in list_new_extracts_nc])

        for month in months:

            print('Working: %s F%03d'%(month, forecast_hour))

            month_files = sorted([f for f in list_new_extracts_nc if month in f])

            month_data_new = xr.open_mfdataset(month_files, concat_dim='valid')

            # Find existing aggregate if it exists:
            append_to_file = nbm_dir + 'agg/blend.%s.qmd.f%03d.WR.nc'%(month, forecast_hour)

            if os.path.isfile(append_to_file):

                append_to_ds = xr.open_dataset(append_to_file)

                existing_valid_times = append_to_ds.valid.values
                new_valid_times = month_data_new.valid.values

                sel_new_valid_times = [t for t in new_valid_times if t not in existing_valid_times]
                
                if len(sel_new_valid_times) > 0:

                    save_out_new = xr.concat(
                            [append_to_ds, month_data_new.sel(valid=sel_new_valid_times)], 
                        dim='valid')

                    # Manually fix the encoding issues...
                    save_out_new['init'].encoding['units'] = 'hours since 2000-01-01 00:00:00'
                    save_out_new['valid'].encoding['units'] = 'hours since 2000-01-01 00:00:00'

                    save_out_new.to_netcdf(append_to_file + '.new')
                    del save_out_new

                    append_to_ds.close()
                    del append_to_ds, month_data_new
                    gc.collect()

                    os.remove(append_to_file)
                    shutil.move(append_to_file + '.new', append_to_file)
                    
                else:
                    append_to_ds.close()
                    del append_to_ds, month_data_new
                    gc.collect()

            else:        
                # Manually fix encoding issues...
                month_data_new['init'].encoding['units'] = 'hours since 2000-01-01 00:00:00'
                month_data_new['valid'].encoding['units'] = 'hours since 2000-01-01 00:00:00'

                month_data_new.to_netcdf(append_to_file)

                del month_data_new
                gc.collect()
                
        [os.remove(f) for f in list_new_extracts_nc]
