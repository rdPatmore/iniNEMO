import xarray as xr
import config
import matplotlib.pyplot as plt
case='EXP13'
fig, axs = plt.subplots(3, figsize=(6.5,4))

def ri_proportion(case, ax):
    ri = xr.open_dataarray(config.data_path() + '/' + case + 
                          '/richardson_number.nc',
                          chunks=dict(time_counter=1))
    area = xr.open_dataset(config.data_path() + '/' + case + 
                           '/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').area
    area = area.isel(x=slice(2,-1), y=slice(2,-1))
    area_sum = area.sum(['x','y'])
    
    ri = ri.sel(deptht=10, method='nearest')
    
    s_unstab = xr.where(ri<0, area,0).sum(['x','y']) / area_sum
    d_unstab = xr.where((0<=ri) & (ri<0.25), area, 0).sum(['x','y']) / area_sum
    m_stab = xr.where((0.25<=ri) & (ri<1), area, 0).sum(['x','y']) / area_sum
    stab = xr.where(ri>=1, area, 0).sum(['x','y']) / area_sum
    
    seg0 = stab
    seg1 = seg0 + m_stab
    seg2 = seg1 + d_unstab
    seg3 = seg2 + s_unstab
    
    ax.fill_between(seg0.time_counter, 0, seg0, facecolor='navy',
                            label='Ri>=1')
    ax.fill_between(seg1.time_counter, seg0, seg1, facecolor='royalblue',
                            label='0.25<=Ri<1')
    ax.fill_between(seg2.time_counter, seg1, seg2, facecolor='lightsteelblue',
                            label='0<=Ri<0.25')
    ax.fill_between(seg3.time_counter, seg2, seg3, facecolor='red',
                            label='Ri<0')

ri_proportion('EXP13', axs[0])
ri_proportion('EXP08', axs[0])

axs[2].set_xlabel('time')
for ax in axs:
    ax.set_ylabel('Ri proportion')
    ax.set_ylim(0,1)
    ax.set_xlim(seg3.time_counter.min(),seg3.time_counter.max())

plt.legend(loc='lower left')

plt.savefig(case + '_ri_proportion.png', dpi=600)
