import xarray as xr
import matplotlib.pyplot as plt
import config
import cmocean



# load data 
file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
preamble = config.data_path() + 'EXP10' +  file_id
kwargs = {'chunks':{'time_counter':100} ,'decode_cf':True} 
mld = xr.open_dataset(preamble + 'grid_T.nc', **kwargs).mldr10_3
bg_mod2 = xr.open_dataset(preamble + 'bg_mod2.nc', **kwargs).bg_mod2**0.5
ice = xr.open_dataset(preamble + 'icemod.nc', **kwargs).siconc

# select time
tslice = '2013-01-08 12:00:00'

depth = 10
sel_dict = dict(time_counter=tslice, deptht=depth)
mld  = mld.sel(time_counter=tslice, method='nearest')
ice  = ice.sel(time_counter=tslice, method='nearest')
bg_mod2  = bg_mod2.sel(sel_dict, method='nearest')

# cut boundaries
bounds = dict(x=slice(45,-45), y=slice(45,-45))
mld = mld.isel(bounds)
ice = ice.isel(bounds)
bg_mod2 = bg_mod2.isel(bounds)

# plot
fig, axs = plt.subplots(1,3)
plt.subplots_adjust(bottom=0.3)
p0 = axs[0].pcolor(mld, vmin=0, vmax=120)
p1 = axs[1].pcolor(bg_mod2, vmin=0, vmax=2e-7)
p2 = axs[2].pcolor(ice, cmap=cmocean.cm.ice, vmin=0, vmax=1)
p = [p0,p1,p2]
txt = ['mld', 'bg', 'sea ice']

for i, ax in enumerate(axs):
    # aspect ratio
    ax.set_aspect('equal')

    # colorbars
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.15, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -3.5, txt[i], fontsize=8,
                 rotation=0, transform=cbar.ax.transAxes,
                 va='center', ha='center')

# save
plt.savefig('mld_and_bg_' + tslice.split(' ')[0] +  '.png', dpi=300)
