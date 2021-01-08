import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

ini = xr.open_dataset('../Output/output.init.nc')

# initialise plots
fig = plt.figure(figsize=(6.5, 4.5), dpi=300)

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

vmin=ini.vomecrty.min()
vmax=ini.vomecrty.max()
vmax = np.abs(max(vmin,vmax))
vmin = - vmax

smin = 33.5
smax = 34.8


ini_horiz = ini.isel(time_counter=0, nav_lev=0)
p0 = axs0[0].pcolor(ini_horiz.x, ini_horiz.y, ini_horiz.vozocrtx,
                    vmin=umin, vmax=umax, cmap=plt.cm.RdBu)
p1 = axs0[1].pcolor(ini_horiz.x, ini_horiz.y, ini_horiz.vomecrty,
                    vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)
p2 = axs0[2].pcolor(ini_horiz.x, ini_horiz.y, ini_horiz.votemper)
p3 = axs0[3].pcolor(ini_horiz.x, ini_horiz.y, ini_horiz.vosaline,
                    vmin=smin, vmax=smax)



ini = ini.assign_coords(x=np.arange(0,51), y=np.arange(0,100))
ini_vert = ini.isel(time_counter=0, y=1)
p0 = axs1[0].pcolor(ini_vert.x, -ini_vert.nav_lev, ini_vert.vozocrtx,
                    vmin=umin, vmax=umax, cmap=plt.cm.RdBu)
p1 = axs1[1].pcolor(ini_vert.x, -ini_vert.nav_lev, ini_vert.vomecrty,
                    vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu)
p2 = axs1[2].pcolor(ini_vert.x, -ini_vert.nav_lev, ini_vert.votemper)
p3 = axs1[3].pcolor(ini_vert.x, -ini_vert.nav_lev, ini_vert.vosaline,
                    vmin=smin, vmax=smax)

pos = axs1[0].get_position()
cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
cbar.ax.text(0.5, -4.5, r'u-vel (m/s)', fontsize=10, rotation=0,
             transform=cbar.ax.transAxes, va='bottom', ha='center')

pos = axs1[1].get_position()
cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
cbar.ax.text(0.5, -4.5, r'v-vel (m/s)', fontsize=10, rotation=0,
             transform=cbar.ax.transAxes, va='bottom', ha='center')

pos = axs1[2].get_position()
cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
cbar = fig.colorbar(p2, cax=cbar_ax, orientation='horizontal')
cbar.ax.text(0.5, -4.5, r'Temperature ($^{\circ}$C)', fontsize=10, rotation=0,
             transform=cbar.ax.transAxes, va='bottom', ha='center')

pos = axs1[3].get_position()
cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
cbar = fig.colorbar(p3, cax=cbar_ax, orientation='horizontal')
cbar.ax.text(0.5, -4.5, r'Salinity (psu)', fontsize=10, rotation=0,
             transform=cbar.ax.transAxes, va='bottom', ha='center')

for ax in axs0:
    ax.set_xticks([])
for ax in axs0[1:]:
    ax.set_yticks([])
for ax in axs1[1:]:
    ax.set_yticks([])

for ax in axs1:
    ax.set_xlabel('lon')
axs0[0].set_ylabel('lat')
axs1[0].set_ylabel('depth (m)')

plt.savefig('ini_state.png')
