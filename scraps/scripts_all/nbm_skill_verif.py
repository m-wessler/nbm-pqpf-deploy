import os, gc, sys
import pygrib
import regionmask
import cartopy
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import multiprocessing as mp
import matplotlib.pyplot as plt 
import matplotlib as mpl

from glob import glob
from numpy import trapz
from scipy.integrate import simps
from sklearn.metrics import auc as auc_calc
from functools import partial
from matplotlib import gridspec
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

from verif_config import *
from verif_funcs import *

import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    
    cwa = sys.argv[1]
    os.makedirs(tmp_dir, exist_ok=True)

    extract_dir = nbm_dir + 'extract/'
    extract_flist = sorted(glob(extract_dir + '*'))

    if not os.path.isfile(urma_dir + 'agg/urma_agg.nc'):
        pass 
        #print('URMA aggregate not found')

    else:
        #print('Getting URMA aggregate from file')
        urma = xr.open_dataset(urma_dir + 'agg/urma_agg.nc')['apcp24h_mm']

    urma = urma/25.4
    urma = urma.rename('apcp24h_in')
    lons, lats = urma.lon, urma.lat

    geodir = '../forecast-zones/'
    zones_shapefile = glob(geodir + '*.shp')[0]

    # Read the shapefile
    zones = gpd.read_file(zones_shapefile)

    # Prune to Western Region using TZ
    zones = zones.set_index('TIME_ZONE').loc[['M', 'Mm', 'm', 'MP', 'P']].reset_index()
    cwas = zones.dissolve(by='CWA').reset_index()[['CWA', 'geometry']]
    _cwas = cwas.copy()

    if cwa == 'WESTUS':
        _cwas['CWA'] = 'WESTUS'
        _cwas = _cwas.dissolve(by='CWA').reset_index()
        bounds = _cwas.total_bounds
    else:
        bounds = _cwas[_cwas['CWA'] == cwa].bounds.values[0]

    print(bounds)

    lons, lats = urma.lon, urma.lat
    mask = regionmask.mask_3D_geopandas(_cwas, lons, lats).rename({'region':'cwa'})
    mask['cwa'] = _cwas.iloc[mask.cwa]['CWA'].values.astype(str)
    mask = mask.sel(cwa=cwa)

    idx = np.where(
        (urma.lat >= bounds[1]) & (urma.lat <= bounds[3]) &
        (urma.lon >= bounds[0]) & (urma.lon <= bounds[2]))

    mask = mask.isel(y=slice(idx[0].min(), idx[0].max()), x=slice(idx[1].min(), idx[1].max()))
    urma = urma.isel(y=slice(idx[0].min(), idx[0].max()), x=slice(idx[1].min(), idx[1].max()))
    urma = urma.transpose('valid', 'y', 'x')

    fhrs = np.arange(fhr_start, fhr_end+1, fhr_step)
    
    extract_pqpf_verif_stats_mp = partial(extract_pqpf_verif_stats, 
                                          _urma=urma, _idx=idx, _mask=mask)

    pool_mod = 2 if cwa == 'WESTUS' else 1
    with mp.get_context('fork').Pool(int(len(fhrs)/pool_mod)) as p:
        returns = p.map(extract_pqpf_verif_stats_mp, fhrs, chunksize=pool_mod)
        p.close()
        p.join()

    returns = [r for r in returns if r is not None]
    scores = xr.concat(returns, dim='fhr')

    stats_dir = '/uufs/chpc.utah.edu/common/home/u1070830/code/nbm-verify/archive/'
    os.makedirs(stats_dir, exist_ok=True)
#     scores.to_netcdf(stats_dir + '%s_stats.nc'%(cwa))

#     scores = xr.open_dataset(stats_dir + '%s_stats.nc'%(cwa))

#     roc_ts = []
#     roc_curve = []

#     for i in range(len(scores.thresh)):

#         _roc_ts = []
#         _roc_curve = []

#         for fhr in scores.fhr:

#             _scores = scores.isel(thresh=i).sel(fhr=fhr)

#             roc_x, roc_y = _scores['false_alarm_rate'], _scores['hit_rate']
#             roc_x = np.append(np.append(1, roc_x), 0)
#             roc_y = np.append(np.append(1, roc_y), 0)

#             roc_lab = _scores.center
#             roc_lab = np.append(np.append(1, roc_lab), 0)

#             _roc_curve.append([roc_x, roc_y, roc_lab])

#             auc = auc_calc(roc_x, roc_y)
#             roc_ss = 2 * (auc - 0.5)

#             _roc_ts.append([auc, roc_ss])

#         roc_curve.append(_roc_curve)
#         roc_ts.append(_roc_ts)

#     roc_curve = np.array(roc_curve)
#     roc_ts = np.array(roc_ts)

#     fig, ax = plt.subplots(1, facecolor='w', figsize=(12, 6))

#     for i, thresh in enumerate(produce_thresholds):

#         ax.plot(scores.fhr, roc_ts[i, :, 1],
#                 marker='x', markersize=10, linewidth=2,
#                 label='> %.2f"'%thresh)

#     ax.axhline(0, color='k')
#     ax.set_xticks(scores.fhr)
#     ax.set_ylabel('ROC Skill Score\n')

#     axx = ax.twinx()
#     ax.set_yticks(ax.get_yticks())
#     axx.set_yticks(ax.get_yticks())
#     axx.set_yticklabels(['%.1f'%v for v in ax.get_yticks()/2 + 0.5])
#     axx.set_ylabel('\nArea Under Curve (AUC)')

#     ax.set_xlabel('\nForecast Hour/Lead Time')

#     date0, date1 = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
#     ax.set_title((
#         'NBM Relative Operating Characteristic | CWA: %s\n'%cwa +
#         '%s - %s\n'%(date0, date1) + 
#         '%02dh Acc QPF\n\n'%(interval) +
#         'Probability of Exceeding Threshold\n'))

#     ax.grid()
#     ax.legend(loc='lower left')

#     savedir = fig_dir + '%s/%s/roc/'%(ver, cwa)
#     os.makedirs(savedir, exist_ok=True)

#     savestr = 'nbm{}_{}_roc_leadtime.png'.format(ver, cwa)

#     plt.savefig(savedir + savestr, dpi=200)
#     print('Saved: ', savestr)

#     plt.close()

#     for i, thresh in enumerate(produce_thresholds):

#         fig, ax = plt.subplots(figsize=(10, 10), facecolor='w')
#         shades = np.linspace(.05, .65, len(scores.fhr))

#         for ii, fhr in enumerate(scores.fhr):

#             ax.plot(roc_curve[i, ii, 0, :], roc_curve[i, ii, 1, :],
#                     linewidth=1, label='F%03d'%fhr, color=str(shades[i]))

#         ax.plot(roc_curve[i, :, 0, :].mean(axis=0), roc_curve[i, :, 1, :].mean(axis=0), 
#                 marker='o', markersize=7.5, color='r', linewidth=2)

#         for point in roc_curve[i].mean(axis=0).T:
#             x, y, s = point
#             ax.text(x*1.04, y*.995, '%.02f'%s, fontsize=10)

#         ax.plot(np.arange(0, 1.1), np.arange(0, 1.1), 'k--')
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)

#         date0, date1 = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
#         ax.set_title((
#             'NBM Relative Operating Characteristic | CWA: %s\n'%cwa +
#             '%s - %s\n'%(date0, date1) + 
#             '%02dh Acc QPF | %3dh Lead Time\n\n'%(interval, fhr) +
#             'Probability of Exceeding %.2f"\n'%thresh))

#         ax.set_xlabel('False Alarm Rate (POFD)')
#         ax.set_ylabel('Probability of Detection (POD)')
#         ax.text(.45, .42, 'No Skill', rotation=45, fontsize=14)

#         ax.text(.812, .055, 'ROCSS: %.2f'%roc_ts[i].mean(axis=0)[1], 
#                 rotation=0, fontsize=14, weight='bold')

#         ax.text(.85, .025, 'AUC: %.2f'%roc_ts[i].mean(axis=0)[0],
#                 rotation=0, fontsize=14, weight='bold')

#         ax.grid()
#         ax.legend(loc='center right')

#         savedir = fig_dir + '%s/%s/roc/'%(ver, cwa)
#         os.makedirs(savedir, exist_ok=True)

#         savestr = 'nbm{}_{}_roc_curve_threshold{}.png'.format(ver, cwa, ('%.2f'%thresh).replace('.', 'p'))

#         plt.savefig(savedir + savestr, dpi=200)
#         print('Saved: ', savestr)

#         plt.close()

#     fig, ax = plt.subplots(1, facecolor='w', figsize=(12, 6))

#     for i, thresh in enumerate(produce_thresholds):

#         ax.plot(scores.fhr, scores['brier_skill'].isel(thresh=i),
#                 marker='x', markersize=10, linewidth=2,
#                 label='> %.2f"'%thresh)

#     ax.axhline(0, color='k')
#     ax.set_xticks(scores.fhr)
#     ax.set_xlabel('\nForecast Hour/Lead Time')

#     date0, date1 = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
#     ax.set_title((
#         'NBM Brier Skill Score | CWA: %s\n'%cwa +
#         '%s - %s\n'%(date0, date1) + 
#         '%02dh Acc QPF\n\n'%(interval) +
#         'Probability of Exceeding Threshold\n'))

#     ax.grid()
#     ax.legend(loc='upper right')

#     savedir = fig_dir + '%s/%s/bss/'%(ver, cwa)
#     os.makedirs(savedir, exist_ok=True)

#     savestr = 'nbm{}_{}_bss_leadtime.png'.format(ver, cwa)

#     plt.savefig(savedir + savestr, dpi=200)
#     print('Saved: ', savestr)

#     plt.close()

#     fig, ax = plt.subplots(1, facecolor='w', figsize=(12, 6))

#     for i, thresh in enumerate(produce_thresholds):

#         ax.plot(scores.fhr, scores['ets'].isel(thresh=i).mean(dim='center'),
#                 marker='x', markersize=10, linewidth=2,
#                 label='> %.2f"'%thresh)

#     ax.axhline(0, color='k')
#     ax.set_xticks(scores.fhr)
#     ax.set_xlabel('\nForecast Hour/Lead Time')

#     date0, date1 = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
#     ax.set_title((
#         'NBM Equitable Threat Score | CWA: %s\n'%cwa +
#         '%s - %s\n'%(date0, date1) + 
#         '%02dh Acc QPF\n\n'%(interval) +
#         'Probability of Exceeding Threshold\n'))

#     ax.grid()
#     ax.legend(loc='upper right')

#     savedir = fig_dir + '%s/%s/ets/'%(ver, cwa)
#     os.makedirs(savedir, exist_ok=True)

#     savestr = 'nbm{}_{}_ets_leadtime.png'.format(ver, cwa)

#     plt.savefig(savedir + savestr, dpi=200)
#     print('Saved: ', savestr)

#     plt.close()