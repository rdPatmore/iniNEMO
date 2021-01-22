import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
ini=0
if ini:
    ini = xr.open_dataset('../Output/output.init.10.nc')
    ini = ini.rename({'nav_lat': 'latt', 'nav_lon': 'lont'})
    time_step=0
else:
    init = xr.open_dataset('../Output/SOCHIC_PATCH_20ts_20150101_20150131_grid_T.nc')
    iniu = xr.open_dataset('../Output/SOCHIC_PATCH_20ts_20150101_20150131_grid_U.nc')
    iniv = xr.open_dataset('../Output/SOCHIC_PATCH_20ts_20150101_20150131_grid_V.nc')
    init = init.rename({'nav_lat': 'latt', 'nav_lon': 'lont',
                        'deptht':'nav_lev'})
    iniu = iniu.rename({'nav_lat': 'latu', 'nav_lon': 'lonu',
                        'depthu':'nav_lev'})
    iniv = iniv.rename({'nav_lat': 'latv', 'nav_lon': 'lonv',
                        'depthv':'nav_lev'})
    ini = xr.merge([init,iniu,iniv])
    time_step=400

# initialise plots
fig = plt.figure(figsize=(6.5, 4.5), dpi=300)
fig.suptitle('time: ' + str(ini.time_counter.values[0]))

# initialise gridspec
gs0 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
gs1 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)

gs0.update(top=0.97, bottom=0.6, left=0.13)
gs1.update(top=0.57, bottom=0.25, left=0.13)

g = 9.81
alpha = -3.2861e-5
beta = 7.8358e-4
axs0, axs1 = [], []
for i in range(4):
    axs0.append(fig.add_subplot(gs0[i]))
    axs1.append(fig.add_subplot(gs1[i]))

umin=ini.vozocrtx.min()
umax=ini.vozocrtx.max()
umax = np.abs(max(umin,umax))
umin = - umax
ulev = np.linspace(umin,umax,11)

vmin=ini.vomecrty.min()
vmax=ini.vomecrty.max()
vmax = np.abs(max(vmin,vmax))
vmin = - vmax
vlev = np.linspace(vmin,vmax,11)

smin = 33.5
smax = 34.8
slev = np.linspace(smin,smax,11)


# plot surface
ini_horiz = ini.isel(time_counter=time_step, nav_lev=0)
p0 = axs0[0].contourf(ini_horiz.lonu, ini_horiz.latu, ini_horiz.vozocrtx,
                    levels=ulev, cmap=plt.cm.RdBu)
p1 = axs0[1].contourf(ini_horiz.lonv, ini_horiz.latv, ini_horiz.vomecrty,
                    levels=vlev, cmap=plt.cm.RdBu)
p2 = axs0[2].contourf(ini_horiz.lont, ini_horiz.latt, ini_horiz.votemper)
p3 = axs0[3].contourf(ini_horiz.lont, ini_horiz.latt, ini_horiz.vosaline,
                    levels=slev)

# plot vertical slice
#ini = ini.assign_coords(x=np.arange(0,51), y=np.arange(0,100))
ini_vert = ini.isel(time_counter=time_step, y=1)
p0 = axs1[0].contourf(ini_vert.lonu, -ini_vert.nav_lev, ini_vert.vozocrtx,
                    levels=ulev, cmap=plt.cm.RdBu)
p1 = axs1[1].contourf(ini_vert.lonv, -ini_vert.nav_lev, ini_vert.vomecrty,
                    levels=vlev, cmap=plt.cm.RdBu)
p2 = axs1[2].contourf(ini_vert.lont, -ini_vert.nav_lev, ini_vert.votemper)
p3 = axs1[3].contourf(ini_vert.lont, -ini_vert.nav_lev, ini_vert.vosaline,
                    levels=slev)

# assign colour bar properties
p = [p0,p1,p2,p3]
labels = [r'u (m/s)',
          r'v (m/s)',
          r'Temperature ($^{\circ}$C)',
          r'Salinity (psu)']

# add colour bars
for i, ax in enumerate(axs1):
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
    cbar.locator = ticker.MaxNLocator(nbins=3)
    cbar.update_ticks()
    cbar.ax.text(0.5, -4.5, labels[i], fontsize=8, rotation=0,
                 transform=cbar.ax.transAxes, va='bottom', ha='center')

for ax in axs0:
    ax.set_xticks([])
for ax in axs0[1:]:
    ax.set_yticks([])
for ax in axs1:
    ax.set_xticks([-2,0,2])
for ax in axs1[1:]:
    ax.set_yticks([])

for ax in axs1:
    ax.set_xlabel('lon')
axs0[0].set_ylabel('lat')
axs1[0].set_ylabel('depth (m)')

plt.savefig('state_t' + str(time_step) + '.png')
