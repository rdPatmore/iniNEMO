import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import config
import cmocean

matplotlib.rcParams.update({'font.size': 8})

# load data 
file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
preamble = config.data_path() + 'EXP10' +  file_id
kwargs = {'chunks':{'time_counter':100} ,'decode_cf':True} 
mld = xr.open_dataset(preamble + 'grid_T.nc', **kwargs).mldr10_3
bg_mod2 = xr.open_dataset(preamble + 'bg_mod2.nc', **kwargs).bg_mod2**0.5
ice = xr.open_dataset(preamble + 'icemod.nc', **kwargs).siconc

# select time
tslice = '2012-12-24 12:00:00'

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
fig, axs = plt.subplots(1,3, figsize=(6.5,3.0))
plt.subplots_adjust(bottom=0.3, top=0.99, right=0.98, left=0.1, hspace=0.05)
p0 = axs[0].pcolor(mld.nav_lon, mld.nav_lat, mld, vmin=0, vmax=120,
                   shading='nearest')
p1 = axs[1].pcolor(bg_mod2.nav_lon, bg_mod2.nav_lat, bg_mod2, vmin=0, vmax=2e-7,
                   shading='nearest')
p2 = axs[2].pcolor(ice.nav_lon, ice.nav_lat, ice, cmap=cmocean.cm.ice,
                   vmin=0, vmax=1, shading='nearest')
p = [p0,p1,p2]
txt = ['Mixed Layer Depth (m)',
        r'$|\nabla b|$ (s$^{-2}$)',
        'Sea Ice Concentration (-)']

for i, ax in enumerate(axs):
    # aspect ratio
    ax.set_aspect('equal')

    # colorbars
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.15, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -5.0, txt[i], fontsize=8,
                 rotation=0, transform=cbar.ax.transAxes,
                 va='center', ha='center')

# set axis labels
axs[0].set_ylabel(r'Latitude ($^{\circ}$N)')
for ax in axs:
    ax.set_xlabel(r'Longitude ($^{\circ}$E)')
for ax in axs[1:]:
    ax.set_yticklabels([])

# save
plt.savefig('mld_and_bg_' + tslice.split(' ')[0] +  '.png', dpi=300)
