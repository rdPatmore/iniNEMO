import xarray as xr
import config
import matplotlib.pyplot as plt
case='EXP13'
fig, axs = plt.subplots(3, figsize=(6.5,5))
plt.subplots_adjust(bottom=0.2)

def ri_proportion(case, ax, depth):
    ri = xr.open_dataarray(config.data_path() + '/' + case + 
                          '/richardson_number.nc',
                          chunks=dict(time_counter=1))
    area = xr.open_dataset(config.data_path() + '/' + case + 
                           '/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').area
    area = area.isel(x=slice(2,-1), y=slice(2,-1))
    print (area)
    area_sum = area.sum(['x','y'])
    
    ri = ri.sel(deptht=depth, method='nearest')
    
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
    ax.legend(loc='lower left', fontsize=6)
    return seg3.time_counter

def add_si_presence(case, ax):
    ''' add presence of sea ice '''
    si_pres = xr.open_dataset(config.data_path() + '/' + case +
                       '/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc').icepres
    si_pres_mean = si_pres.mean(['x','y'])
    ax.plot(si_pres.time_counter, si_pres_mean)

def add_surface_heat_flux(case, ax):
    ''' add presence of sea ice '''
    si_pres = xr.open_dataset(config.data_path() + '/' + case +
                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').qt_oce
    si_pres_mean = si_pres.mean(['x','y'])
    si_pres_quantile = si_pres.quantile([0.1,0.9],['x','y'])
    ax.plot(si_pres.time_counter, si_pres_mean)
    ax.fill_between(si_pres.time_counter,
                    si_pres_quantile.sel(quantile=0.1),
                    si_pres_quantile.sel(quantile=0.9),
                    facecolor='navy', alpha=0.2)
    ax.set_ylabel('surface heat flux')
    ax.axhline(0, ls='--', lw=0.8, c='black')

def add_smoothed_wind_speed(case, ax):
    ''' add presence of sea ice '''
    wind = xr.open_dataset(config.data_path() + '/' + case +
                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').windsp
    wind = wind.mean(['x','y']).rolling(time_counter=8,
                                                   min_periods=1).mean()
    ax.set_ylabel('wind speed')
    ax.plot(wind.time_counter, wind)

depth=10
t=ri_proportion('EXP13', axs[0], depth)
add_surface_heat_flux('EXP13', axs[1])
add_smoothed_wind_speed('EXP13', axs[2])
#t = ri_proportion('EXP08', axs[1])
#add_surface_heat_flux('EXP08', axs[1])
#add_smoothed_wind_speed('EXP08', axs[1])

axs[2].set_xlabel('time')
axs[0].set_ylabel('Ri proportion')
axs[0].set_ylim(0,1)
for ax in axs:
    ax.set_xlim(t.min(),t.max())
axs[0].set_xticks([])
axs[1].set_xticks([])
for label in axs[2].get_xticklabels():
  label.set_rotation(20)
  label.set_ha('right')


plt.savefig(case + '_ri_proportion_' + str(depth) + '.png', dpi=600)
