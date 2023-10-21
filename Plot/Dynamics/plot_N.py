import config
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
case = 'EXP13'

#wind = xr.open_dataset(config.data_path() + '/' + case +
#                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').windsp
#si_pres = xr.open_dataset(config.data_path() + '/' + case +
#                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').qt_oce
n2 = xr.open_dataset(config.data_path() + '/' + case +
                     '/SOCHIC_PATCH_3h_20121209_20130331_grid_W.nc',
                    ).bn2
                    #chunks=dict(time_counter=100)).bn2

fig = plt.figure()#constrained_layout=True)

gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1:, 0])
ax3 = fig.add_subplot(gs[1:, 1])
ax4 = fig.add_subplot(gs[1:, 2])
gs.update(right=0.83)

n2 = n2.isel(x=slice(20,-20), y=slice(20,-20))
n2_prof = n2.sel(depthw=slice(0,100)).mean(['x','y'], skipna=True)
print (n2_prof.min().values)
print (n2_prof.max().values)
p0 = ax1.pcolor(n2_prof.time_counter, -n2_prof.depthw, n2_prof.T,
                cmap=plt.cm.magma_r, vmin=0,vmax=1e-5)
n2_surf = n2.isel(depthw=1)

def render_slice(n2, ax, time, m_ax):
    n2_t = n2.isel(time_counter=time)
    print (n2_t.min().values)
    print (n2_t.max().values)
    ax.pcolor(n2_t.nav_lon, n2_t.nav_lat, n2_t, vmin=0, vmax=1e-5,
              cmap=plt.cm.magma_r)
    ax.set_aspect('equal')
    m_ax.axvline(n2_t.time_counter.values, lw=0.8, c='black')
                 #n2.depthw.min().values,
                 #n2.depthw.max().values)

render_slice(n2_surf, ax2, 100, ax1)
render_slice(n2_surf, ax3, 300, ax1)
render_slice(n2_surf, ax4, 500, ax1)

pos0 = ax1.get_position()
pos1 = ax3.get_position()
cbar_ax = fig.add_axes([0.85, pos1.y0, 0.02, pos0.y1 - pos1.y0])
cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
#cbar.locator = ticker.MaxNLocator(nbins=3)
#cbar.update_ticks()
cbar.ax.text(4.5, 0.5, r'$N^2$', fontsize=8, rotation=0,
             transform=cbar.ax.transAxes, va='center', ha='left')
    
ax1.set_xlabel('time')
ax1.set_ylabel('depth')
for label in ax1.get_xticklabels():
  label.set_rotation(20)
  label.set_ha('right')

ax2.set_ylabel('latitude')
ax3.set_yticks([])
ax4.set_yticks([])
for ax in [ax2, ax3, ax4]:
    ax.set_xlabel('longitude')

plt.savefig(case + '_N2_sat.png', dpi=300)


