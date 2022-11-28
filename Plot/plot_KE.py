import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case):
        self.case = case
        file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
        self.preamble = config.data_path() + case + '/' + file_id

    def plot_ke_time_series(self):
        ''' plot ke oce-ice over depth and time '''
 
        # set axes
        fig, axs = plt.subplots(3, figsize=(6.5,5))
        plt.subplots_adjust(right=0.88, top=0.98, left=0.1)

        # load data 
        ds = xr.open_dataset(self.preamble + 'TKE_oce_ice.nc')
        ds = ds.isel(z=slice(0,20))

        # render
        vmin, vmax = 0, 0.011
        p0 = axs[0].pcolor(ds.time_counter, -ds.z,
                           ds.TKE_oce.T, vmin=vmin, vmax=vmax,
                           shading='nearest')
        p1 = axs[1].pcolor(ds.time_counter, -ds.z,
                           ds.TKE_ice.T, vmin=vmin, vmax=vmax,
                           shading='nearest')

        bg = xr.open_dataset(self.preamble + 'bg_norm_ice_oce_quantile.nc')
        bg_ice = bg.bg_norm_ice
        bg_oce = bg.bg_norm_oce
        axs[2].fill_between(bg_ice.time_counter,
                                 bg_ice.sel(quantile=0.05),
                                 bg_ice.sel(quantile=0.95),
                                 alpha=0.4, color='navy', edgecolor=None)
        axs[2].fill_between(bg_oce.time_counter,
                                 bg_oce.sel(quantile=0.05),
                                 bg_oce.sel(quantile=0.95),
                                 alpha=0.4, color='orange', edgecolor=None)
        axs[2].plot(bg_ice.time_counter, bg_ice.sel(quantile=0.5), c='navy',
                    label='sea ice')
        axs[2].plot(bg_oce.time_counter, bg_oce.sel(quantile=0.5), c='orange',
                    label='ocean')

        pos0 = axs[0].get_position()
        pos1 = axs[1].get_position()
        cbar_ax = fig.add_axes([0.89, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        txt = r'TKE [m$^2$ s$^{-2}]$'
        cbar.ax.text(4.3, 0.5, txt, fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left')

        axs[2].legend(loc='upper right', title='partition',
                       bbox_to_anchor=(0.5, 0.90, 0.5, 0.1), fontsize=6)

        date_lims = (bg.time_counter.min(), 
                     bg.time_counter.max())
        
        for ax in axs:
            ax.set_xlim(date_lims)
        for ax in axs[:-1]:
            ax.set_xticklabels([])

        axs[0].set_ylabel('Depth [m]')
        axs[1].set_ylabel('Depth [m]')
        axs[2].set_ylabel(r'$|\nabla b|$ [s$^{-2}]$')

        # date labels
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        axs[-1].set_xlabel('date')
       
        axs[0].text(0.98, 0.95, 'open ocean', transform=axs[0].transAxes,
                    va='top', ha='right')
        axs[1].text(0.98, 0.95, 'sea ice', transform=axs[1].transAxes,
                    va='top', ha='right')

        # save
        plt.savefig('ke_oce_ice.png', dpi=600)

    
ke = plot_KE('EXP10')
ke.plot_ke_time_series()
