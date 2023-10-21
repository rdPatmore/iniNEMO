import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cmocean

ds = xr.open_dataset('../InterpZ/ini_state.nc')

ds = ds.isel(X=slice(1,-2), Y=slice(1,-2), Z=slice(0,-1))

t0 = ds.votemper.isel(X=0).values[:,::-1]
t1 = ds.votemper.isel(Y=0).values
t2 = ds.votemper.isel(Y=-1).values[:,::-1]
t3 = ds.votemper.isel(X=-1).values

s0 = ds.vosaline.isel(X=0).values[:,::-1]
s1 = ds.vosaline.isel(Y=0).values
s2 = ds.vosaline.isel(Y=-1).values[:,::-1]
s3 = ds.vosaline.isel(X=-1).values

t = np.concatenate((t0,t1,t3,t2), axis=1)
s = np.concatenate((s0,s1,s3,s2), axis=1)

fig, axs = plt.subplots(2,1, figsize=(6.5,3))
plt.subplots_adjust(left=0.15, right=0.81)

lev0 = np.linspace(-2,1.2,17)
p0 = axs[0].contourf(np.arange(t.shape[1]), -ds.depth, t, cmap=plt.cm.plasma_r,
                     levels=lev0)
lev1 = np.linspace(34.1,34.75,27)
p1 = axs[1].contourf(np.arange(s.shape[1]), -ds.depth, s,cmap=cmocean.cm.haline,
                     levels=lev1)

pos = axs[0].get_position()
cbar_ax = fig.add_axes([0.83, pos.y0, 0.02, pos.y1 - pos.y0])
cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
cbar.ax.text(6.3, 0.5, r'Temperature ($^{\circ}$C)', fontsize=10, rotation=90,
             transform=cbar.ax.transAxes, va='center', ha='left')

pos = axs[1].get_position()
cbar_ax = fig.add_axes([0.83, pos.y0, 0.02, pos.y1 - pos.y0])
cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
cbar.ax.text(6.3, 0.5, r'Salinity (psu)', fontsize=10, rotation=90,
             transform=cbar.ax.transAxes, va='center', ha='left')

for ax in axs:
    ax.set_xticks([])
    ax.set_ylabel('depth (m)')

plt.savefig('ini_conditions.png', dpi=1200)
