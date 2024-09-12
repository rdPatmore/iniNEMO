import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean
import numpy as np

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case):
        self.case = case
        file_id = 'SOCHIC_PATCH_{}_20121209_20121211_'
        self.path = config.data_path() + case + '/'
        self.preamble = config.data_path() + case + '/ProcessedVars/' + file_id

    def plot_compare_time_sampling(self):
        ''' plot the TKE budget for three time sampling rates '''
        # ini figure
        fig, axs = plt.subplots(3, 8, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.12,
                            wspace=0.05, hspace=0.30)

        # get data
        ds_15 = xr.open_dataset(self.preamble.format('15mi') + 
                                'TKE_budget_z_integ.nc', chunks='auto')
        ds_30 = xr.open_dataset(self.preamble.format('30mi') + 
                                'TKE_budget_z_integ.nc', chunks='auto')
        #ds_60 = xr.open_dataset(self.preamble.format('60mi') + 
        #                        'TKE_budget_z_integ.nc', chunks='auto')

        # set cbar range and render
        self.vmin, self.vmax = -1e-5, 1e-5
        self.render_horizontal_slice(ds_15, axs[0])
        p = self.render_horizontal_slice(ds_30, axs[1])
        #self.render_horizontal_slice(ds_60, axs[:,2])

        # titles
        titles = ['Horiz. Pressure\nGradient',
                  'Lateral\nAdvection ',
                  'Vertical\nAdvection',
                  'Vertical Diffusion',
                  'Ice-Ocean Drag',
                  'Wind Stress',
                  'Vertical Buoyancy\nFlux',
                  'Tendency' ]

        for i, ax in enumerate(axs[0].flatten()):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal')

        # format axes
        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel(r'Longitude ($^{\circ}$E)')
        for ax in axs[:,0].flatten():
            ax.set_ylabel(r'Latitude ($^{\circ}$N)')

        # add colour bar
        pos0 = axs[0,-1].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([0.88, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'EKE Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        fn = '_tke_budget_depth_integrated_sampling_rate_compare.png'
        plt.savefig(self.case + fn, dpi=600)

#    def plot_compare_time_sampling_anomalies(self):
#        ''' plot the TKE budget anomalies for three time sampling rates '''

    def plot_ml_integrated_TKE_budget(self):
        ''' plot budget of TKE depth-integrated over the mixed layer '''
        
        # load and slice
        ds = xr.open_dataset(self.preamble + 'TKE_budget_z_integ.nc')
        
        self.vmin, self.vmax = -1e-5, 1e-5
        self.render_horizontal_slice(ds)

        plt.savefig(self.case + '_tke_budget_depth_integrated.png', dpi=600)

    def render_horizontal_slice(self, ds, axs):

        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # trim edges
        cut=slice(10,-10)
        ds     = ds.isel(x=cut,y=cut)
        cfg    = cfg.isel(x=cut,y=cut)

        ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
        ds['trd_hpg'] = ds.trd_hpg
        ds['trd_tot'] = ds.trd_tot

        # plot
        self.cmap=cmocean.cm.balance
        def render(ax, ds, var):
            p = ax.pcolor(cfg.nav_lon, cfg.nav_lat, ds[var],
                      vmin=self.vmin, vmax=self.vmax,
                      cmap=self.cmap)
            return p

        # where is ldf ???
        render(axs[0], ds, 'trd_hpg')
        render(axs[1], ds, 'trd_adv')
        render(axs[2], ds, 'trd_zad')
        render(axs[3], ds, 'trd_zdf')
        render(axs[4], ds, 'trd_tfr2d')
        render(axs[5], ds, 'trd_tau2d')
        render(axs[6], ds, 'trd_bfx')
        p = render(axs[7], ds, 'trd_tot')

        return p

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
        plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.19)

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
        axs.set_ylabel(r'EKE (m$^2$s$^{-3})$')

        plt.savefig(self.case + '_tke_budget_domain_integrated_zoned_new.png',
                    dpi=600)

    def plot_laterally_integrated_TKE_budget_ice_oce_zones(self):
        ''' plot laterally integrated TKE budget for each ice ocean zone'''
     
        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.13, right=0.82, top=0.93, bottom=0.15)

        def get_ds_and_combine_vars(zone):
            ds = xr.open_dataset(
                 self.preamble + 'TKE_budget_horizontal_integ_' + zone + '.nc')

            ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
            ds['trd_hpg'] = ds.trd_hpg

            return ds

        ds_miz = get_ds_and_combine_vars('miz')
        ds_ice = get_ds_and_combine_vars('ice')
        ds_oce = get_ds_and_combine_vars('oce')

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
                ax.plot(da, da.deptht, label=titles[i], lw=1.0)
            var_sum = ds.trd_hpg + ds.trd_adv + ds.trd_zad \
                    + ds.trd_zdf + ds.trd_tfr2d + ds.trd_tau2d \
                    + ds.trd_bfx
            print (var_sum)
            ax.plot(var_sum, var_sum.deptht, label='sum', lw=0.5)

        # render miz
        render_depth_budget(axs[0], ds_miz, var_list, titles)
        axs[0].set_title('Marginal Ice Zone')

        # render ice
        render_depth_budget(axs[1], ds_ice, var_list, titles)
        axs[1].set_title('Sea Ice Zone')

        # render oce
        render_depth_budget(axs[2], ds_oce, var_list, titles)
        axs[2].set_title('Open Ocean')

        for ax in axs:
            # set limits
            ax.invert_yaxis()
            ax.set_ylim(50,0)
            ax.set_xlim(-3e-8,3e-8)

            # set axis labels
            ax.set_xlabel(r'EKE (m$^2$s$^{-3}$)')
        axs[0].set_ylabel('Depth (m)')

        # remove y labels
        for ax in axs[1:]:
            ax.set_yticklabels([])

        # plot legend 
        axs[2].legend(bbox_to_anchor=[1.01,1])

        plt.savefig(self.case + '_tke_budget_horiz_integrated_zoned.png',
                    dpi=600)

    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_15mi_20121209_20121211_'
ke = plot_KE('TRD00')
#ke.plot_domain_integrated_TKE_budget_ice_oce_zones()
#ke.plot_laterally_integrated_TKE_budget_ice_oce_zones()
#ke.plot_ke_time_series()
#ke.plot_ml_integrated_TKE_budget()
ke.plot_compare_time_sampling()
#print ('depth integrated - done')
#ke.plot_z_slice_TKE_budget(depth=10)
print ('depth slice - done')
#ke.plot_domain_integrated_TKE_budget()
