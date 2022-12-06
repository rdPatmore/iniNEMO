import config
import xarray as xr
import matplotlib.pyplot as plt
import glidertools as gt
from plot_interpolated_tracks import get_sampled_path
import numpy as np
import cartopy.crs as ccrs


# domain projection
axes_proj=ccrs.AlbersEqualArea(central_latitude=-60,
                                standard_parallels=(-62,-58))

#axins.spines['geo'].set_visible(False)
#axins.patch.set_alpha(0.0)

#path = config.data_path() + 'EXP10/GliderRandomSampling/' 
#giddy = xr.open_dataset(path + 'glider_uniform_interp_1000_01.nc')
#giddy = giddy.isel(ctd_depth=10)
#giddy['lon_offset'] = giddy.attrs['lon_offset']
#giddy['lat_offset'] = giddy.attrs['lat_offset']
#giddy = giddy.set_coords(['lon_offset','lat_offset','time_counter'])
#print (giddy)
#giddy = get_transects(giddy.votemper, offset=True, cut_meso=False)
##giddy = xr.open_dataset(path + 'glider_uniform_interp_1000_pre_transect_00.nc')
##giddy = giddy.stack(z=['ctd_depth','distance'])
#giddy = giddy.where(giddy.meso_transect!=0, drop=True)
##giddy = xr.where(giddy.meso_transect==0, giddy, np.nan).dropna('distance')

fig = plt.figure(figsize=(3.2, 4))
ax0 = fig.add_subplot(211,projection=axes_proj)
ax1 = fig.add_subplot(212)
c1 = '#f18b00'
path_cset=[c1,'navy','turquoise','purple']

# plot path
proj = ccrs.PlateCarree() # lon lat projection
glider_data = get_sampled_path('EXP10', 'interp_1000',
                               post_transect=True, drop_meso=True) 
for i, (l,trans) in enumerate(glider_data.groupby('transect')):
    ax0.plot(trans.lon, trans.lat, transform=proj, 
             c=path_cset[int(trans.vertex[0])], lw=0.5)
#axs[0].plot(giddy.lon, giddy.lat)

giddy_raw = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
giddy_raw['distance'] = xr.DataArray( 
                         gt.utils.distance(giddy_raw.longitude,
                                           giddy_raw.latitude).cumsum(),
                                           dims='ctd_data_point')
giddy_raw = giddy_raw.where((giddy_raw.dives<50) &
                            (giddy_raw.dives>=41), drop=True)
print (giddy_raw)
ax1.plot(giddy_raw.distance.diff('ctd_data_point').cumsum()/1000,
            -giddy_raw.isel(ctd_data_point=slice(None,-1)).ctd_depth)

plt.savefig('giddy_path.png', dpi=600)
