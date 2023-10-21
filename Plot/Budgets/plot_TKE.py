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

    def plot_TKE_budget_30(self):
        ''' plot budget of TKE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1,
                            wspace=0.10, hspace=0.11)
        axs[-1,-1].axis('off')

        # load and slice
        ds = xr.open_dataset(self.preamble + 'TKE_30_budget.nc')
        b_flux = xr.open_dataarray(self.preamble + 'b_flux_rey.nc')
        b_flux = b_flux.sel(deptht=30, method='nearest')
        b_flux = b_flux.drop(['nav_lon','nav_lat'])

        cut=slice(10,-10)
        ds     = ds.isel(x=cut,y=cut)
        b_flux = b_flux.isel(x=cut,y=cut)

        ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
        print (ds.trd_hpg)
        print (b_flux)
        ds['trd_hpg'] = ds.trd_hpg + b_flux

        # plot
        self.vmin, self.vmax = -2e-7, 2e-7
        self.cmap=cmocean.cm.balance
        def render(ax, ds, var):
            ax.pcolor(ds[var],
                      vmin=self.vmin, vmax=self.vmax,
                      cmap=self.cmap)

        render(axs[0,0], ds, 'trd_hpg')
        render(axs[0,1], ds, 'trd_adv')
        render(axs[0,2], ds, 'trd_pvo')
        render(axs[0,3], ds, 'trd_zad')
        render(axs[1,0], ds, 'trd_zdf')
        p = axs[1,1].pcolor(-b_flux, vmin=self.vmin, vmax=self.vmax,
                        cmap=self.cmap)
        render(axs[1,2], ds, 'trd_tot')

        # titles

        # titles
        titles = ['pressure grad',
                  'lateral\n advection ',
                  'Coriolis',               'vertical\nadvection',
                  'vertical\ndiffusion','vertical\nbuoyancy flux',
                  'tendency' ]

        for i, ax in enumerate(axs.flatten()[:-1]):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal')

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel('x')
        for ax in axs[:,0].flatten():
            ax.set_ylabel('y')

        pos0 = axs[0,-1].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([0.86, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'TKE Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        plt.savefig(self.case + '_tke_30_budget.png')



    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
ke = plot_KE('TRD00', file_id)
ke.plot_TKE_budget_30()
