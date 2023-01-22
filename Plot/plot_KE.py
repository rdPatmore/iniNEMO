import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case, file_id):
        self.case = case
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

    def plot_KE_budget_slices(self):
        ''' plot budget of KE '''
        
        # ini figure
        fig, axs = plt.subplots(4, 4, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'KE_24_mean.nc')
        ds = ds.sel(deptht=30, method='nearest') # depth

        # 0.5 * d(u'_bar^2)/dt
        trd_tot = (ds.KE.diff('time_counter') /\
                  ds.time_counter.astype('float64').diff('time_counter')).squeeze()*1e9
        print (trd_tot)
 

        ds = ds.isel(time_counter=1)             # time

        # plot
        vmin, vmax = -5e-3, 5e-3
        cmap=cmocean.cm.balance
        axs[0,0].pcolor(ds.ketrd_hpg, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,1].pcolor(ds.ketrd_spg, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,2].pcolor(ds.ketrd_keg, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,3].pcolor(ds.ketrd_rvo, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,0].pcolor(ds.ketrd_pvo, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,1].pcolor(ds.ketrd_zad, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[1,2].pcolor(ds.ketrd_udx, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[1,3].pcolor(ds.ketrd_ldf, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2,0].pcolor(ds.ketrd_zdf, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2,1].pcolor(ds.ketrd_tau, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[2,2].pcolor(ds.ketrd_bfr, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[2,3].pcolor(ds.ketrd_bfri, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3,0].pcolor(ds.ketrd_atf, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3,1].pcolor(ds.ketrd_convP2K, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[3,3].pcolor(trd_tot, vmin=vmin, vmax=vmax, cmap=cmap)

        # sum
        kesum = ds.ketrd_hpg + ds.ketrd_spg + ds.ketrd_keg + ds.ketrd_rvo +  \
                ds.ketrd_pvo + ds.ketrd_zad + \
                ds.ketrd_zdf + ds.ketrd_tau + \
                ds.ketrd_atf + ds.ketrd_convP2K
        axs[3,2].pcolor(kesum, vmin=vmin, vmax=vmax, cmap=cmap)

        # residule
        resid = kesum - trd_tot
        #axs[3,3].pcolor(resid, vmin=vmin, vmax=vmax, cmap=cmap)


        # titles
        axs[0,0].set_title('hyd p')
        axs[0,1].set_title('surf p')
        axs[0,2].set_title('hadv')
        axs[0,3].set_title('zeta')
        axs[1,0].set_title('cori')
        axs[1,1].set_title('vadv')
        axs[1,2].set_title('udx')
        axs[1,3].set_title('hdiff')
        axs[2,0].set_title('zdiff')
        axs[2,1].set_title('wind')
        axs[2,2].set_title('imdrag')
        axs[2,3].set_title('expdrag')
        axs[3,0].set_title('asselin')
        axs[3,1].set_title('PE2KE')
        axs[3,2].set_title('sum of terms')
        axs[3,3].set_title('TOT')
        #axs[3,4].set_title('residule')

        plt.savefig(self.case + '_ke_mld_budget.png')

    def plot_TKE_budget_mld(self):
        ''' plot budget of TKE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(3, 4, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'tke_budget.nc')

        # plot
        vmin, vmax = -1e-7, 1e-7
        cmap=cmocean.cm.balance
        axs[0,0].pcolor(ds.trd_hpg, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,1].pcolor(ds.trd_spg, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,2].pcolor(ds.trd_keg, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,3].pcolor(ds.trd_rvo, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,0].pcolor(ds.trd_pvo, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,1].pcolor(ds.trd_zad, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,2].pcolor(ds.trd_zdf, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,3].pcolor(ds.trd_atf, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2,0].pcolor(ds.trd_tot, vmin=vmin, vmax=vmax, cmap=cmap)

        # sum
        kesum = ds.trd_hpg + ds.trd_spg + ds.trd_keg + ds.trd_rvo +  \
                ds.trd_pvo + ds.trd_zad# + \
                #ds.trd_zdf 
        axs[2,1].pcolor(kesum, vmin=vmin, vmax=vmax, cmap=cmap)

        # residule
        resid = kesum - ds.trd_tot
        axs[2,2].pcolor(resid, vmin=vmin, vmax=vmax, cmap=cmap)


        # titles
        axs[0,0].set_title('hyd p')
        axs[0,1].set_title('surf p')
        axs[0,2].set_title('ke adv')
        axs[0,3].set_title('zeta adv')
        axs[1,0].set_title('cori')
        axs[1,1].set_title('v adv')
        axs[1,2].set_title('zdiff')
        axs[1,3].set_title('asselin')
        axs[2,0].set_title('TOT')
        axs[2,1].set_title('sum of terms')
        axs[2,2].set_title('residule')

        plt.savefig(self.case + '_tke_mld_budget.png')


    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
ke = plot_KE('EXP90', file_id)
ke.plot_TKE_budget_mld()
