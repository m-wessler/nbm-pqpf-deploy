def extract_pqpf_verif_stats(_fhr, _urma):

    nbm_file = glob(nbm_dir + 'extract/nbm_probx_fhr%03d.nc'%_fhr)[0]
    print(nbm_file)
    
    # Subset the threshold value
    nbm = xr.open_dataset(nbm_file)['probx'].sel(
    y=slice(idx[0].min(), idx[0].max()),
    x=slice(idx[1].min(), idx[1].max()))

    # Subset the times
    nbm_time = nbm.valid
    urma_time = _urma.valid
    time_match = nbm_time[np.in1d(nbm_time, urma_time)].values
    time_match = np.array([t for t in time_match if pd.to_datetime(t) >= start_date])
    time_match = np.array([t for t in time_match if pd.to_datetime(t) <= end_date])
    date0 = pd.to_datetime(time_match[0]).strftime('%Y/%m/%d %H UTC')
    date1 = pd.to_datetime(time_match[-1]).strftime('%Y/%m/%d %H UTC')

    _nbm = nbm.sel(valid=time_match)
    _urma = _urma.sel(valid=time_match)
    nbm_mask, _nbm = xr.broadcast(mask, _nbm)
    urma_mask, _urma = xr.broadcast(mask, _urma)

    _nbm_masked = xr.where(nbm_mask, _nbm, np.nan)
    _urma_masked = xr.where(urma_mask, _urma, np.nan)
        
    data = []
    
    for thresh in produce_thresholds:
        
        print('Processing f%03d %.2f"'%(_fhr, thresh))

        _nbm_masked_select = _nbm_masked.sel(threshold=thresh)/100
        
        bins = np.arange(0, 101, 10)

        N = xr.where(~np.isnan(_nbm_masked_select), 1, 0).sum()
        n = xr.where(_urma_masked > thresh, 1, 0).sum()
        o = n/N
        uncertainty = o * (1 - o)
        
        reliability_inner = []
        resolution_inner = []
        reliability_diagram = []
        roc_diagram = []

        for i, bounds in enumerate(zip(bins[:-1], bins[1:])):

            left, right = np.array(bounds)/100
            center = round(np.mean([left, right]), 2)
            
            fk = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right), _nbm_masked_select, np.nan)
            nk = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right), 1, 0).sum()

            ok_count = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right) & (_urma_masked > thresh), 1, 0).sum()
            ok = ok_count/nk
            
            #        3D          1D     3D   1D
            _reliability_inner = nk * ((fk - ok)**2)
            _reliability_inner['center'] = center
            reliability_inner.append(_reliability_inner)

            #        1D         1D     1D   1D
            _resolution_inner = nk * ((ok - o)**2)
            _resolution_inner['center'] = center
            resolution_inner.append(_resolution_inner)
            
            reliability_diagram.append([center, ok.values])
                        
            hit = xr.where((_nbm_masked_select > center) & (_urma_masked > thresh), 1, 0).sum(dim='valid')
            false_alarm = xr.where((_nbm_masked_select > center) & (_urma_masked <= thresh), 1, 0).sum(dim='valid')
            
            observed_yes = xr.where(_urma_masked > thresh, 1, 0).sum(dim='valid')
            observed_no = xr.where(_urma_masked <= thresh, 1, 0).sum(dim='valid')
            
            hit_rate = hit/observed_yes
            false_alarm_rate = false_alarm/observed_no
            
            roc_diagram.append([false_alarm_rate.mean().values, hit_rate.mean().values, center])
        
        reliability_inner = xr.concat(reliability_inner, dim='center')
        reliability_inner_condensed = xr.where(mask, reliability_inner.sum(dim='center'), np.nan)
        
        reliability = (1/N) * reliability_inner_condensed
        reliability = reliability.mean(dim='valid')
        
        resolution = (1/N) * xr.concat(resolution_inner, dim='center').sum(dim='center')
        
        brier = reliability - resolution + uncertainty
        brier_score = brier.mean().values
        
        brier_skill = 1 - (brier/o)
        brier_skill_score = brier_skill.mean().values
        
        brier = brier.rename('brier')
        brier_skill = brier_skill.rename('brier_skill')
        
        reliability_diagram = np.array(reliability_diagram).T
        roc_diagram = np.array(roc_diagram).T
        
        far = xr.DataArray(roc_diagram[0], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        hr = xr.DataArray(roc_diagram[1], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})
        
        data_merge = xr.merge([brier_skill])

        # Need to figure out reliability scaling and add in here as (x, y)
        data_merge['n_events'] = observed_yes        
        data_merge['hit_rate'] = hr
        data_merge['false_alarm_rate'] = far
                
        data.append(data_merge)
                                        
    return xr.concat(data, dim='thresh')


##################


from scipy.integrate import simps
from sklearn.metrics import auc as auc_calc
from sklearn.metrics import roc_curve

def extract_pqpf_verif_stats(_fhr, _urma):

    nbm_file = glob(nbm_dir + 'extract/nbm_probx_fhr%03d.nc'%_fhr)[0]
    print(nbm_file)
    
    # Subset the threshold value
    nbm = xr.open_dataset(nbm_file)['probx'].sel(
    y=slice(idx[0].min(), idx[0].max()),
    x=slice(idx[1].min(), idx[1].max()))

    # Subset the times
    nbm_time = nbm.valid
    urma_time = _urma.valid
    time_match = nbm_time[np.in1d(nbm_time, urma_time)].values
    time_match = np.array([t for t in time_match if pd.to_datetime(t) >= start_date])
    time_match = np.array([t for t in time_match if pd.to_datetime(t) <= end_date])
    date0 = pd.to_datetime(time_match[0]).strftime('%Y/%m/%d %H UTC')
    date1 = pd.to_datetime(time_match[-1]).strftime('%Y/%m/%d %H UTC')

    _nbm = nbm.sel(valid=time_match)
    _urma = _urma.sel(valid=time_match)
    nbm_mask, _nbm = xr.broadcast(mask, _nbm)
    urma_mask, _urma = xr.broadcast(mask, _urma)

    _nbm_masked = xr.where(nbm_mask, _nbm, np.nan)
    _urma_masked = xr.where(urma_mask, _urma, np.nan)
        
    data = []
    
    for thresh in produce_thresholds[:3]:
        
        print('Processing f%03d %.2f"'%(_fhr, thresh))

        _nbm_masked_select = _nbm_masked.sel(threshold=thresh)/100
        
        bins = np.arange(0, 101, 10)

        N = xr.where(~np.isnan(_nbm_masked_select), 1, 0).sum()
        n = xr.where(_urma_masked > thresh, 1, 0).sum()
        o = n/N
        uncertainty = o * (1 - o)
        
        reliability_inner = []
        resolution_inner = []
        reliability_diagram = []
        roc_diagram = []#[[1., 1., 1.]]

        for i, bounds in enumerate(zip(bins[:-1], bins[1:])):

            left, right = np.array(bounds)/100
            center = round(np.mean([left, right]), 2)
            
            fk = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right), _nbm_masked_select, np.nan)
            nk = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right), 1, 0).sum()

            ok_count = xr.where((_nbm_masked_select > left) & (_nbm_masked_select <= right) & (_urma_masked > thresh), 1, 0).sum()
            ok = ok_count/nk
            
            #        3D          1D     3D   1D
            _reliability_inner = nk * ((fk - ok)**2)
            _reliability_inner['center'] = center
            reliability_inner.append(_reliability_inner)

            #        1D         1D     1D   1D
            _resolution_inner = nk * ((ok - o)**2)
            _resolution_inner['center'] = center
            resolution_inner.append(_resolution_inner)
            
            #print(_fhr, thresh, center, np.round(ok.values, 3))
            reliability_diagram.append([center, ok.values])
                        
            hit = xr.where((_nbm_masked_select > center) & (_urma_masked > thresh), 1, 0).sum(dim='valid')
            false_alarm = xr.where((_nbm_masked_select > center) & (_urma_masked <= thresh), 1, 0).sum(dim='valid')
            
            observed_yes = xr.where(_urma_masked > thresh, 1, 0).sum(dim='valid')
            observed_no = xr.where(_urma_masked <= thresh, 1, 0).sum(dim='valid')
            
#             print(center)
#             print(hit.mean().values, observed_yes.mean().values)
#             print(false_alarm.mean().values, observed_no.mean().values)
#             print()
            
            hit_rate = hit/observed_yes
            false_alarm_rate = false_alarm/observed_no
            
            roc_diagram.append([false_alarm_rate.mean().values, hit_rate.mean().values, center])
            
        #roc_diagram.append([0., 0., 0.])

        reliability_inner = xr.concat(reliability_inner, dim='center').sum(dim='center')
        reliability_inner = xr.where(mask, reliability_inner, np.nan)
        
        reliability = (1/N) * reliability_inner
        reliability = reliability.mean(dim='valid')
        #print('\nreli:', reliability.mean().values)
        
        resolution = (1/N) * xr.concat(resolution_inner, dim='center').sum(dim='center')
        #print('reso:', resolution.values)
        
        brier = reliability - resolution + uncertainty
        brier_score = brier.mean().values
        #print('brier:', brier_score)
        
        brier_skill = 1 - (brier/o)
        brier_skill_score = brier_skill.mean().values
        
        brier = brier.rename('brier')
        brier_skill = brier_skill.rename('brier_skill')
        #print('brier skill:', brier_skill.mean().values)
        
        reliability_diagram = np.array(reliability_diagram).T
        roc_diagram = np.array(roc_diagram).T
        
        auc = auc_calc(roc_diagram[0], roc_diagram[1])
        roc_ss = 2 * (auc - 0.5)
        print(auc, roc_ss)
        
#         plt.figure(facecolor='w')
#         plt.plot(roc_diagram[0], roc_diagram[1], 'k-^')
#         plt.xlim([0, 1])
#         plt.ylim([0, 1])
#         for x, y, s in zip(roc_diagram[0], roc_diagram[1], roc_diagram[2]):
#             plt.text(x, y, s)
#         plt.title('%s %s %s'%(cwa, _fhr, thresh))
#         plt.xlabel('False Alarm Rate')
#         plt.ylabel('Hit Rate')
#         plt.plot(np.arange(0, 1.1, .1), np.arange(0, 1.1, .1), 'k--', linewidth=0.5)
#         plt.grid()
#         plt.show()
    
        far = xr.DataArray(roc_diagram[0], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})#, name='false_alarm_rate'),
        hr = xr.DataArray(roc_diagram[1], dims={'center':roc_diagram[2]}, coords={'center':roc_diagram[2]})#, name='hit_rate')
        
        data_merge = xr.merge([brier_skill])
        
        data_merge['n_events'] = observed_yes
        data_merge['hit_rate'] = hr
        data_merge['false_alarm_rate'] = far
        
#         data_merge['auc'] = auc
#         data_merge['roc'] = roc_ss

        print(data_merge)
        
        data.append(data_merge)
                                
    return xr.concat(data, dim='thresh')

verif_stats = extract_pqpf_verif_stats(48, urma)