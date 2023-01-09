import config
import xarray as xr
import matplotlib.pyplot as plt
import glidertools as gt
from plot_interpolated_tracks import get_sampled_path
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'font.size': 8})

                     # ~~~ horizontal path ~~~ #

# projections
axes_proj=ccrs.AlbersEqualArea(central_latitude=-60,
                                standard_parallels=(-62,-58))
proj = ccrs.PlateCarree() # lon lat projection
print (axes_proj)

# initialise figure
fig = plt.figure(figsize=(5.5, 2.0))
gs0 = gridspec.GridSpec(ncols=1, nrows=1)
gs1 = gridspec.GridSpec(ncols=1, nrows=1)

gs0.update(top=0.87, bottom=0.2, left=0.70, right=0.99)
gs1.update(top=0.87, bottom=0.2, left=0.12, right=0.57)

ax0 = fig.add_subplot(gs0[0])
ax1 = fig.add_subplot(gs1[0])
    
# get data
glider_data = get_sampled_path('EXP10', 'interp_1000',
                               post_transect=True, drop_meso=True) 

# plot path
path_cset=['#f18b00','navy','green','purple']
#path_cset=['#f18b00','navy','turquoise','purple']
for i, (l,trans) in enumerate(glider_data.groupby('transect')):
    ax0.plot(trans.lon - trans.lon_offset, trans.lat - trans.lat_offset,
             c=path_cset[int(trans.vertex[0])], lw=0.5)

ax0.set_aspect(2.0)
ax0.set_xlabel(r'Longitude ($^{\circ}$)')
ax0.set_ylabel(r'Latitude ($^{\circ}$)')
ax0.set_title('\'Bow-Tie\'', fontsize=8)

ax0.text(0.02, 1.01, '(b)', transform=ax0.transAxes, ha='left', va='bottom',
         fontsize=8)
ax1.text(0.02, 1.01, '(a)', transform=ax1.transAxes, ha='left', va='bottom',
         fontsize=8)
#print (glider_data.lon.min().values - glider_data.lon_offset)
#print (glider_data.lon.max().values - glider_data.lon_offset)
#print (glider_data.lat.min().values - glider_data.lat_offset)
#print (glider_data.lat.max().values - glider_data.lat_offset)
#ax0.set_xticks(np.linspace( -0.2,   0.2, 5), crs=proj)
#ax0.set_yticks(np.linspace(-60.1, -59.9, 5), crs=proj)
#lon_formatter = LongitudeFormatter(zero_direction_label=True)
#lat_formatter = LatitudeFormatter()
#ax0.xaxis.set_major_formatter(lon_formatter)
#ax0.yaxis.set_major_formatter(lat_formatter)
#xticks = np.linspace( -0.2,   0.2, 5)
#yticks = np.linspace(-60.1, -59.9, 5)
#gl = ax0.gridlines(xlocs=xticks, ylocs=yticks, draw_labels=True)
#gl.xlines=False
#gl.ylines=False
#gl.xlabel_top=False
#gl.ylabel_right=False
##print (LongitudeFormatter.__doc__)
#lon_formatter = LongitudeFormatter(number_format='.1f',
#                                   degree_symbol='')
###                                   #direction_label=False)
#lat_formatter = LatitudeFormatter(number_format='.1f',
#                                  degree_symbol='')
##                                  #direction_label=False)
##ax0.xaxis.set_major_formatter(lon_formatter)
##ax0.yaxis.set_major_formatter(lat_formatter)
#
##gl.ylocator = mticker.FixedLocator(yticks)
##gl.xlocator = mticker.FixedLocator(xticks)
#ax0.tick_params(axis='both',labelsize=6,direction='out',right=False,top=False)
#:wqax0.axes.spines['geo'].xaxis.set_tick_params(direction="out", length=8)

                       # ~~~ depth path ~~~ #

# plot path
giddy_raw = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
giddy_raw['distance'] = xr.DataArray( 
                         gt.utils.distance(giddy_raw.longitude,
                                           giddy_raw.latitude).cumsum(),
                                           dims='ctd_data_point')
giddy_raw = giddy_raw.where((giddy_raw.dives<50) &
                            (giddy_raw.dives>=41), drop=True)
print (giddy_raw)
ax1.plot(giddy_raw.distance.diff('ctd_data_point').cumsum()/1000,
        -giddy_raw.isel(ctd_data_point=slice(None,-1)).ctd_depth,
        c=path_cset[0])
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Depth (m)')
ax1.set_title('Dive Pattern', fontsize=8)

#ax0.spines['right'].set_visible(False)
#ax1.spines['right'].set_visible(False)
#ax0.spines['top'].set_visible(False)
#ax1.spines['top'].set_visible(False)

plt.savefig('giddy_path.png', dpi=600)
