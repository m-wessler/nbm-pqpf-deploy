# Most of this was written in the first iteration of the code, so has not gone through the same
# level of vetting as the final products. However, there are some useful functions in here
# for polling the Synoptic API for data that are still very much valid.

import os
import csv
import requests

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import urllib.request as req
import scipy.stats as scipy

from datetime import datetime, timedelta

# Colorblind-friendly Palete
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
          "#0072B2", "#D55E00", "#CC79A7", "#999999"]

_site, _date0, _date1 = None, None, None
_init_hours, _dates = None, None
_datadir, _figdir = None, None
_thresholds, _thresh_id = None, None

def get_1d_csv(get_req, this, total, verbose=True):
    
    _date, _init_hour, _url = get_req
    
    try:
        response = req.urlopen(_url).read().decode('utf-8')
        if verbose:
            print('\r[%d/%d] %s %s'%(this, total, _date, _init_hour), end='')
        
    except:
        if verbose:
            print('\r[%d/%d] NOT FOUND %s %s'%(this, total, _date, _init_hour), end='')
        return None
    
    else:
        init = datetime(_date.year, _date.month, _date.day, _init_hour, 0)

        response = response.split('\n')
        header = np.append('InitTime', response[0].split(','))
        
        lines = []
        for line in response[1:]:
            line = line.split(',')

            try:
                line[0] = datetime.strptime(line[0], '%Y%m%d%H')
            except:
                pass
            else:
                lines.append(np.append(init, line))
                        
        return header, lines
    

def get_precip_obs(s, d0, d1, verbose=False):

    # Tokens registered to michael.wessler@noaa.gov
    api_token = 'a2386b75ecbc4c2784db1270695dde73'
    api_key = 'Kyyki2tc1ETHUgShiscycW15e1XI02SzRXTYG28Dpg'
    base = 'https://api.synopticdata.com/v2/stations/precip?'
    
    allints = []
    
    forecast_interval = 6
    for interval in [6, 12, 24]:
        
        # Limit how big the observation lag can be (minutes)
        lag_limit = (interval/2)*60
        repeat = int((interval-forecast_interval)/6)+1
        
        df = []
        while repeat > 0:
            if verbose:
                print('Working: Interval {}h Iteration {}'.format(interval, repeat))
                        
            _d0 = d0+timedelta(hours=(forecast_interval)*(repeat-1))
            _d1 = d1+timedelta(hours=1+forecast_interval*(repeat-1))
            
            url = base + 'stid={}&start={}&end={}&pmode=intervals&interval={}&token={}'.format(
                s,
                datetime.strftime(_d0, '%Y%m%d%H%M'),
                datetime.strftime(_d1, '%Y%m%d%H%M'),
                interval, api_token)
                        
            api_data_raw = requests.get(url).json()

            vdates = pd.date_range(_d0, _d1, freq='%dh'%interval)
            
            for i in api_data_raw['STATION'][0]['OBSERVATIONS']['precipitation']:
                
                if i['last_report'] is not None:
                    
                    try:
                        last_rep = datetime.strptime(i['last_report'], '%Y-%m-%dT%H:%M:%SZ')
                        vtime = vdates[np.argmin(np.abs(vdates - last_rep))]
                        lag_mins = (vtime - last_rep).seconds/60
                        value = float(i['total']) if lag_mins < lag_limit else np.nan
                    except:
                        #raise
                        pass
                    else:
                        #print('{}\t{}\t{}\t{}'.format(vtime, last_rep, lag_mins, value))
                        df.append([vtime, lag_mins, value])
                    
            repeat -= 1

        df = pd.DataFrame(df, 
            columns=['ValidTime', '%sh_lag_mins'%interval, '%sh_precip_mm'%interval]
            ).set_index('ValidTime').sort_index()       
        
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='6H'))
        df.index.rename('ValidTime', inplace=True)
                
        allints.append(df)

    return allints

def apcp_dist_plot(_obs, _nbm, _interval, trimZero=True, show=False):
    
    iobs = _obs['%dh_precip_in'%_interval]
    #iobs[iobs <= 0.01] = np.nan
    iobs = iobs[iobs > 0.01] if trimZero else iobs
    
    ifx = _nbm['APCP%dhr_surface'%_interval]
    #ifx[ifx <= 0.01] = np.nan
    ifx = ifx[ifx > 0.01] if trimZero else ifx
    
    _threshold = np.append(0, np.nanpercentile(iobs, (33, 67, 100)))
    #_threshold = np.array([np.ceil(x*10)/10 for x in _threshold])
    
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), facecolor='w')
    ax, axx = axs
    
    maxval = max(np.nanmax(iobs), np.nanmax(iobs))
    binsize = 0.05
    
    ax.hist(iobs, bins=np.arange(0, maxval, binsize), 
            edgecolor='k', density=True, color=colors[0], alpha=0.75,
            label='Observed PDF (%.2f in bins)'%binsize)
    
    ax.hist(ifx, bins=np.arange(0, maxval, binsize), 
        edgecolor='k', density=True, color=colors[1], alpha=0.50,
        label='Forecast PDF (%.2f in bins)'%binsize)
    
    axx.hist(iobs, bins=np.arange(0, maxval, 0.00001), 
            density=True, cumulative=True, histtype='step', 
            linewidth=2.5, edgecolor=colors[0])
    axx.plot(0, linewidth=2.5, color=colors[0], label='Observed CDF (Continuous)')
    
    axx.hist(ifx, bins=np.arange(0, maxval, 0.00001), 
            density=True, cumulative=True, histtype='step', 
            linewidth=2.5, linestyle='-', edgecolor=colors[1])
    axx.plot(0, linewidth=2.5, linestyle='-', color=colors[1], label='Forecast CDF (Continuous)')
    
    for p, c in zip([33, 67], [colors[4], colors[5]]):
        ax.axvline(np.nanpercentile(iobs, p), color=c, linewidth=3, zorder=100, 
                   label='%dth Percentile _obs: %.2f in'%(p, np.nanpercentile(iobs, p)))
    
    axx.set_ylabel('\nCumulative [%]')
    axx.set_yticks([0, .2, .4, .6, .8, 1.0])
    axx.set_yticklabels([0, 20, 40, 60, 80, 100])
    axx.set_ylim([0, 1.01])
        
    for axi in axs:
        axi.set_xticks(np.arange(0, maxval+binsize, binsize*2))
        axi.set_xticklabels(['%.2f'%v for v in np.arange(0, maxval+binsize, binsize*2)], rotation=45)
        axi.set_xlim([0, maxval-binsize])
        axi.set_xlabel('\n%dh Forecast Precipitation [in]'%_interval)
        
        axi.set_ylabel('Frequency [%]\n')
        
        axi.set_title('%s\n%dh Forecast Precipitation\nNBM v3.2 Period %s – %s\n'%(
            _site, _interval, _date0.strftime('%Y-%m-%d'), _date1.strftime('%Y-%m-%d')))
        
        axi.grid(True)
        
    ax.legend(loc='upper right')
    axx.legend(loc='lower right')
  
    plt.tight_layout()
    
    savestr = '{}_{}h.APCP_dist.png'.format(_site, _interval)
    
    os.makedirs(_figdir + 'apcp_dist/', exist_ok=True)
    plt.savefig(_figdir + 'apcp_dist/' + savestr, dpi=150)
    
    if show:
        plt.show()
    else:
        print(savestr)
        plt.close()
    
    return _threshold
        
def get_nbm_1d_mp(_stid, verbose=False):
    
    nbmfile = _datadir + '%s_nbm_%s_%s.pd'%(_stid, _date0.strftime('%Y%m%d'), _date1.strftime('%Y%m%d'))
    
    if os.path.isfile(nbmfile):
        # Load file
        #_nbm = pd.read_pickle(nbmfile)
        if verbose:
            # print('Loaded NBM from file %s'%nbmfile)
            print('NBM file exists%s\n'%nbmfile)
        else:
            pass

    else:
        url_list = []
        for date in _dates:
            for init_hour in _init_hours:
                # For now pull from the csv generator
                # Best to get API access or store locally later
                base = 'https://hwp-viz.gsd.esrl.noaa.gov/wave1d/data/archive/'
                datestr = '{:04d}/{:02d}/{:02d}'.format(date.year, date.month, date.day)
                sitestr = '/NBM/{:02d}/{:s}.csv'.format(init_hour, _stid)
                url_list.append([date, init_hour, base + datestr + sitestr])
        
        # Try multiprocessing this for speed?
        _nbm = np.array([get_1d_csv(url, this=i+1, total=len(url_list),
                                    verbose=False) for i, url in enumerate(url_list)])
        _nbm = np.array([line for line in _nbm if line is not None])

        try:
            header = _nbm[0, 0]
            
        except:
            if verbose:
                print('No NBM 1D Flat File for %s\n'%_stid)
            _nbm = (_stid, None)
            nbmfile = None
            
        else:
            if verbose:
                print('Producing NBM data from 1D Flat File for %s\n'%_stid)
            # This drops days with incomplete collections. There may be some use
            # to keeping this data, can fix in the future if need be
            # May also want to make the 100 value flexible!
            _nbm = np.array([np.array(line[1]) for line in _nbm if len(line[1]) == 100])

            _nbm = _nbm.reshape(-1, _nbm.shape[-1])
            _nbm[np.where(_nbm == '')] = np.nan

            # Aggregate to a clean dataframe
            _nbm = pd.DataFrame(_nbm, columns=header).set_index(
                ['InitTime', 'ValidTime']).sort_index()

            # Drop last column (misc metadata?)
            _nbm = _nbm.iloc[:, :-2].astype(float)
            header = _nbm.columns

            # variables = np.unique([k.split('_')[0] for k in header])
            # levels = np.unique([k.split('_')[1] for k in header])

            init =  _nbm.index.get_level_values(0)
            valid = _nbm.index.get_level_values(1)

#             # Note the 1h 'fudge factor' in the lead time here
#             lead = pd.DataFrame(
#                 np.transpose([init, valid, ((valid - init).values/3600/1e9).astype(int)+1]), 
#                 columns=['InitTime', 'ValidTime', 'LeadTime']).set_index(['InitTime', 'ValidTime'])

#             _nbm.insert(0, 'LeadTime', lead)

            klist = np.array([k for k in np.unique([k for k in list(_nbm.keys())]) if ('APCP' in k)&('1hr' not in k)])
            klist = klist[np.argsort(klist)]
#             klist = np.append('LeadTime', klist)
            _nbm = _nbm.loc[:, klist]

#             # Nix values where lead time shorter than acc interval
#             for k in _nbm.keys():
#                 if 'APCP24hr' in k:
#                     _nbm[k][_nbm['LeadTime'] < 24] = np.nan
#                 elif 'APCP12hr' in k:
#                     _nbm[k][_nbm['LeadTime'] < 12] = np.nan
#                 elif 'APCP6hr' in k:
#                     _nbm[k][_nbm['LeadTime'] < 6] = np.nan
#                 else:
#                     pass
                
            _nbm.to_pickle(nbmfile)
        
            if verbose:
                print('\nSaved NBM to file %s\n'%nbmfile)

    return nbmfile

def get_precip_obs_mp(_stid, verbose=False):
        
    obfile = _datadir + '%s_obs_%s_%s.pd'%(_stid, _date0.strftime('%Y%m%d'), _date1.strftime('%Y%m%d'))

    if os.path.isfile(obfile):
        if verbose:
            # print('Loaded obs from file %s'%obfile)
            print('Obs File Exists %s'%obfile)
        else:
            pass

    else:
        # Get and save file
        iobs = get_precip_obs(_stid, _date0, _date2, verbose=False)
        iobs = iobs[0].merge(iobs[1], how='inner', on='ValidTime').merge(iobs[2], how='inner', on='ValidTime')
        iobs = iobs[[k for k in iobs.keys() if 'precip' in k]].sort_index()

        iobs.to_pickle(obfile)
        if verbose:
            print('Saved obs to file %s\n'%obfile)
        del iobs
    
    return obfile
