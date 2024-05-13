import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean
import numpy as np

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case, file_id):
        self.case = case
        self.preamble = config.data_path() + case + '/' + file_id
        self.path = config.data_path() + case + '/'

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
        bg_ice = bg.bg_norm_ice.squeeze()
        bg_oce = bg.bg_norm_oce.squeeze()
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
        print (self.case + '_ke_oce_ice.png')
        plt.savefig(self.case + '_ke_oce_ice.png', dpi=600)

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

    def plot_TKE_budget(self, depth=None, ml_mean=False):
        ''' plot budget of TKE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1,
                            wspace=0.10, hspace=0.11)
        axs[-1,-1].axis('off')

        # load and slice
        ds = xr.open_dataset(self.preamble + 'TKE_budget.nc')
        ds = ds.sel(deptht=depth, method='nearest')
        b_flux = xr.open_dataarray(self.preamble + 'b_flux_rey.nc')
        b_flux = b_flux.sel(deptht=depth, method='nearest')
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

        plt.savefig(self.case + '_tke_' + str(depth) + '_budget.png')

    def plot_ml_integrated_TKE_budget(self):
        ''' plot budget of TKE depth-integrated over the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,3.5))
        plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.12,
                            wspace=0.05, hspace=0.30)
        #axs[-1,-1].axis('off')

        # load and slice
        ds = xr.open_dataset(self.preamble + 'TKE_budget_z_integ.nc')
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        cut=slice(10,-10)
        ds     = ds.isel(x=cut,y=cut)
        cfg    = cfg.isel(x=cut,y=cut)

        ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
        ds['trd_hpg'] = ds.trd_hpg
        ds['trd_tot'] = ds.trd_tot

        # plot
        self.vmin, self.vmax = -1e-5, 1e-5
        self.cmap=cmocean.cm.balance
        def render(ax, ds, var):
            p = ax.pcolor(cfg.nav_lon, cfg.nav_lat, ds[var],
                      vmin=self.vmin, vmax=self.vmax,
                      cmap=self.cmap)
            return p

        render(axs[0,0], ds, 'trd_hpg')
        render(axs[0,1], ds, 'trd_adv')
        render(axs[0,2], ds, 'trd_zad')
        render(axs[0,3], ds, 'trd_zdf')
        render(axs[1,0], ds, 'trd_tfr2d')
        render(axs[1,1], ds, 'trd_tau2d')
        p = render(axs[1,2], ds, 'trd_bfx')
        p = render(axs[1,3], ds, 'trd_tot')

        # titles
        titles = ['Horiz. Pressure\nGradient',
                  'Lateral\nAdvection ',
                  'Vertical\nAdvection',
                  'Vertical Diffusion',
                  'Ice-Ocean Drag',
                  'Wind Stress',
                  'Vertical Buoyancy\nFlux',
                  'Tendency' ]

        for i, ax in enumerate(axs.flatten()):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal')

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel(r'Longitude ($^{\circ}$E)')
        for ax in axs[:,0].flatten():
            ax.set_ylabel(r'Latitude ($^{\circ}$N)')

        pos0 = axs[0,-1].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([0.86, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'TKE Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        plt.savefig(self.case + '_tke_budget_depth_integrated.png', dpi=600)

    def plot_domain_integrated_TKE_budget(self):
        ''' plot domain integrated TKE budget '''
     
        # ini figure
        fig, axs = plt.subplots(1, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.13, right=0.95, top=0.98, bottom=0.19)

        # load and slice
        ds = xr.open_dataset(self.preamble + 'TKE_budget_domain_integ.nc')

        ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
        ds['trd_hpg'] = ds.trd_hpg# + ds.trd_bfx
        #ds['trd_tot'] = -ds.trd_tot

        # plot
        self.vmin, self.vmax = -1e-5, 1e-5
        self.cmap=cmocean.cm.balance

        # titles
        titles = ['Horiz.\nPressure\nGradient',
                  'Lateral\nAdvection ',
                  'Vertical\nAdvection',
                  'Vertical\nDiffusion',
                  'Ice-Ocean\n Drag',
                  'Wind\nStress',
                  'Vertical\nBuoyancy\nFlux',
                  'Tendency' ]

        # set list of terms
        var_list = [
        'trd_hpg',
        'trd_adv',
        'trd_zad',
        'trd_zdf',
        'trd_tfr2d',
        'trd_tau2d',
        'trd_bfx',
        'trd_tot']

        # render data
        data = [ds[var].values for var in var_list]
        axs.bar(titles, data)

        # set axis labels
        axs.set_xlabel('Component')
        axs.set_ylabel('EKE')

        plt.savefig(self.case + '_tke_budget_domain_integrated.png', dpi=600)

    def plot_domain_integrated_TKE_budget_ice_oce_zones(self):
        ''' plot domain integrated TKE budget for each ice ocean zone'''
     
        # ini figure
        fig, axs = plt.subplots(1, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.13, right=0.95, top=0.98, bottom=0.19)

        def get_ds_and_combinde_vars(zone):
            ds = xr.open_dataset(
                     self.preamble + 'TKE_budget_domain_integ_' + zone + '.nc')

            ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
            ds['trd_hpg'] = ds.trd_hpg

            return ds

        ds_miz = get_ds_and_combinde_vars('miz')
        ds_ice = get_ds_and_combinde_vars('ice')
        ds_oce = get_ds_and_combinde_vars('oce')

        # plot
        self.vmin, self.vmax = -1e-5, 1e-5
        self.cmap=cmocean.cm.balance

        # titles
        titles = ['Horiz.\nPressure\nGradient',
                  'Lateral\nAdvection ',
                  'Vertical\nAdvection',
                  'Vertical\nDiffusion',
                  'Ice-Ocean\n Drag',
                  'Wind\nStress',
                  'Vertical\nBuoyancy\nFlux',
                  'Tendency' ]

        # set list of terms
        var_list = [
        'trd_hpg',
        'trd_adv',
        'trd_zad',
        'trd_zdf',
        'trd_tfr2d',
        'trd_tau2d',
        'trd_bfx',
        'trd_tot']

        # render data
        x = np.arange(len(var_list))
        width = 0.25

        # render miz
        data_miz = [ds_miz[var].values for var in var_list]
        axs.bar(x, data_miz, width, label='MIZ')

        # render ice
        data_ice = [ds_ice[var].values for var in var_list]
        axs.bar(x + width, data_ice, width, label='Ice')

        # render oce
        data_oce = [ds_oce[var].values for var in var_list]
        axs.bar(x + width * 2, data_oce, width, label='Oce')

        # set tickes
        axs.set_xticks(x + width, titles)

        # set axis labels
        axs.set_xlabel('Component')
        axs.set_ylabel('EKE')

        plt.savefig(self.case + '_tke_budget_domain_integrated_zoned_new.png',
                    dpi=600)

    def plot_laterally_integrated_TKE_budget_ice_oce_zones(self):
        ''' plot laterally integrated TKE budget for each ice ocean zone'''
     
        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.13, right=0.85, top=0.98, bottom=0.19)

        def get_ds_and_combinde_vars(zone):
            ds = xr.open_dataset(
                 self.preamble + 'TKE_budget_horizontal_integ_' + zone + '.nc')

            ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
            ds['trd_hpg'] = ds.trd_hpg

            return ds

        ds_miz = get_ds_and_combinde_vars('miz')
        ds_ice = get_ds_and_combinde_vars('ice')
        ds_oce = get_ds_and_combinde_vars('oce')

        # titles
        titles = ['Horiz.\nPressure\nGradient',
                  'Lateral\nAdvection ',
                  'Vertical\nAdvection',
                  'Vertical\nDiffusion',
                  'Ice-Ocean\n Drag',
                  'Wind\nStress',
                  'Vertical\nBuoyancy\nFlux',
                  'Tendency' ]

        # set list of terms
        var_list = [
        'trd_hpg',
        'trd_adv',
        'trd_zad',
        'trd_zdf',
        'trd_tfr2d',
        'trd_tau2d',
        'trd_bfx',
        'trd_tot']

        def render_depth_budget(ax, ds, var_list, titles):
            for i, var in enumerate(var_list):
                da = ds[var]
                ax.plot(da, da.deptht, label=titles[i], lw=0.8)
            var_sum = ds.trd_hpg + ds.trd_adv + ds.trd_zad \
                    + ds.trd_zdf + ds.trd_tfr2d + ds.trd_tau2d \
                    + ds.trd_bfx
            print (var_sum)
            ax.plot(var_sum, var_sum.deptht, label='sum', lw=0.5)

        # render miz
        render_depth_budget(axs[0], ds_miz, var_list, titles)

        # render ice
        render_depth_budget(axs[1], ds_ice, var_list, titles)

        # render oce
        render_depth_budget(axs[2], ds_oce, var_list, titles)

        for ax in axs:
            # set limits
            ax.invert_yaxis()
            ax.set_ylim(50,0)
            ax.set_xlim(-3e-8,3e-8)

            # set axis labels
            ax.set_ylabel('EKE')

        # remove y labels
        for ax in axs[1:]:
            ax.set_yticklabels([])

        # plot legend 
        axs[2].legend(bbox_to_anchor=[1.01,1])

        plt.savefig(self.case + '_tke_budget_horiz_integrated_zoned.png',
                    dpi=600)

    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_15mi_20121209_20121211_'
ke = plot_KE('TRD00', file_id)
#ke.plot_domain_integrated_TKE_budget_ice_oce_zones()
ke.plot_laterally_integrated_TKE_budget_ice_oce_zones()
#ke.plot_ke_time_series()
#ke.plot_ml_integrated_TKE_budget()
#ke.plot_domain_integrated_TKE_budget()
