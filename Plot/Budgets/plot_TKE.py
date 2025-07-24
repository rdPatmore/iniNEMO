import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean
import numpy as np
from dask.diagnostics import ProgressBar

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case, file_id):
        self.case = case
        self.preamble = config.data_path() + case + '/' + file_id
        self.proc_preamble = config.data_path() + case + '/ProcessedVars/'\
                             + file_id
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
        
        # load and slice
        ds = xr.open_dataset(self.proc_preamble + 'TKE_budget_z_integ.nc')
        
        self.vmin, self.vmax = -2e-6, 2e-6
        self.render_horizontal_slice(ds)

        plt.savefig(self.case + '_tke_budget_depth_integrated.png', dpi=600)

    def plot_z_slice_TKE_budget(self, depth=10):
        ''' plot budget of TKE depth-integrated over the mixed layer '''
        
        # load and slice
        ds = xr.open_dataset(self.preamble + 'TKE_budget_full.nc')
        ds = ds.sel(deptht=depth, method='nearest')
        
        self.vmin, self.vmax = -2e-7, 2e-7
        self.render_horizontal_slice(ds)

        plt.savefig(self.case + '_tke_budget_depth_{}.png'.format(str(depth)),
                                                                  dpi=600)

    def render_horizontal_slice(self, ds):

        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.1, right=0.85, top=0.90, bottom=0.12,
                            wspace=0.05, hspace=0.30)
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # trim edges
        cut=slice(10,-10)
        ds     = ds.isel(x=cut,y=cut)
        cfg    = cfg.isel(x=cut,y=cut)

        #ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
        ds['trd_adv'] = ds.trd_keg + ds.trd_zad
        ds['trd_hpg'] = ds.trd_hpg
        ds['trd_tot'] = ds.trd_tot
        ds['trd_tau2d'] = ds.trd_tau2d - ds.trd_tfr2d
        # rdp should this not be trd and tau (3d versions)

        # plot
        self.cmap=cmocean.cm.balance
        def render(ax, ds, var):
            p = ax.pcolor(cfg.nav_lon, cfg.nav_lat, ds[var],
                      vmin=self.vmin, vmax=self.vmax,
                      cmap=self.cmap)
            return p

        # where is ldf ???
        render(axs[0,0], ds, 'trd_hpg')
        render(axs[0,1], ds, 'trd_adv')
        render(axs[0,2], ds, 'trd_rvo')
        render(axs[0,3], ds, 'trd_zdf')
        render(axs[1,0], ds, 'trd_tfr2d')
        render(axs[1,1], ds, 'trd_tau2d')
        p = render(axs[1,2], ds, 'trd_bfx')
        p = render(axs[1,3], ds, 'trd_tot')
        # this should be zdf 

        # titles
        titles = ['Horiz. Pressure\nGradient',
                  'Advection',
                  'Barotropic\nInstability',
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
        cbar_ax = fig.add_axes([0.88, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'EKE Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')


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
        plt.subplots_adjust(left=0.13, right=0.85, top=0.95, bottom=0.19)

        def get_ds_and_combinde_vars(zone):
            ds = xr.open_dataset(
                     self.proc_preamble + 'TKE_budget_domain_integ_' + zone + '.nc')

            ds['trd_adv'] = ds.trd_keg + ds.trd_zad
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
                  'Advection ',
                  'Barotropic\nInstability',
                  'Vertical\nDiffusion',
                  'Ice-Ocean\n Drag',
                  'Wind\nStress',
                  'Vertical\nBuoyancy\nFlux',
                  'Tendency' ]

        # set list of terms
        var_list = [
        'trd_hpg',
        'trd_adv',
        'trd_rvo',
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

        # legend
        axs.legend(bbox_to_anchor=[1.01,1])

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
                 self.proc_preamble + 'TKE_budget_horizontal_integ_' + zone + '.nc')

            ds['trd_adv'] = ds.trd_keg + ds.trd_zad
            ds['trd_hpg'] = ds.trd_hpg

            return ds

        ds_miz = get_ds_and_combine_vars('miz')
        ds_ice = get_ds_and_combine_vars('ice')
        ds_oce = get_ds_and_combine_vars('oce')

        # titles
        titles = ['Horiz.\nPressure\nGradient',
                  'Advection ',
                  'Barotropic\nInstability',
                  'Vertical\nDiffusion',
                  'Ice-Ocean\n Drag',
                  'Wind\nStress',
                  'Vertical\nBuoyancy\nFlux',
                  'Tendency' ]

        # set list of terms
        var_list = [
        'trd_hpg',
        'trd_adv',
        'trd_rvo',
        'trd_zdf',
        'trd_tfr2d',
        'trd_tau2d',
        'trd_bfx',
        'trd_tot']

        def render_depth_budget(ax, ds, var_list, titles):
            for i, var in enumerate(var_list):
                da = ds[var]
                ax.plot(da, da.deptht, label=titles[i], lw=1.0)
            var_sum = ds.trd_hpg + ds.trd_adv + ds.trd_rvo \
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

class plot_KE_Tedesco(object):

    def __init__(self, case, file_id):
        self.case = case
        self.preamble = config.data_path() + case + '/' + file_id
        self.proc_preamble = config.data_path() + case + '/ProcessedVars/'\
                             + file_id
        self.path = config.data_path() + case + '/'

    def mask_mld(self, ds):
        ''' mask below mixed layer depth '''

        mld = xr.open_dataset(self.preamble + 'grid_T.nc', chunks="auto"
                                   ).mldr10_3
        mld = self.get_and_trim_ds(self.preamble + "grid_T.nc").mldr10_3
        ds = ds.where(ds.deptht < mld)
        print (ds)

        return ds

    def domain_integral(self, tke_mld):

        # calculate domain integral
        tke_integ = (tke_mld * self.bt).sum()

        return tke_integ

    def depth_integral(self, EKE):

        # depth integral
        if "deptht" in list(EKE.dims.keys()):
            EKE = self.mask_mld(EKE)
            e3t = self.get_and_trim_ds(self.preamble + 'grid_T.nc').e3t
            EKE = (EKE * e3t).sum("deptht")
        else:
            print ("no depth dim")

        return EKE

    def get_and_trim_ds(self, path, slice_vals=slice(10,-10)):

        # get ds
        var = xr.open_dataset(path, chunks="auto")

        # cut rim
        var_cut = var.isel(x=slice(10,-10),y=slice(10,-10))

        return var_cut

    def partition_by_ice_cover(self, eke_mld, threshold=0.2):

        # load ice concentration
        icemsk = self.get_and_trim_ds(self.preamble + "icemod.nc").siconc
        icemsk = icemsk.isel(time_counter=0)
        cfg = self.get_and_trim_ds(self.path + "domain_cfg.nc").squeeze()
        e3t = self.get_and_trim_ds(self.preamble + 'grid_T.nc').e3t

        # get masks
        miz_msk = (icemsk > threshold) & (icemsk < (1 - threshold))
        ice_msk = icemsk > (1 - threshold)
        oce_msk = icemsk < threshold

        # mask by ice concentration
        tke_mld_miz = eke_mld.where(miz_msk)
        tke_mld_ice = eke_mld.where(ice_msk)
        tke_mld_oce = eke_mld.where(oce_msk)

        # find volume of each partition
        area = cfg.e2t * cfg.e1t
        t_vol = area * e3t
        t_vol_miz = t_vol.where(miz_msk).sum()
        t_vol_ice = t_vol.where(ice_msk).sum()
        t_vol_oce = t_vol.where(oce_msk).sum()

        # calculate volume weighted mean
        t_volume = e3t * area
        tke_integ_miz = (tke_mld_miz * t_volume).sum() / t_vol_miz
        tke_integ_ice = (tke_mld_ice * t_volume).sum() / t_vol_ice
        tke_integ_oce = (tke_mld_oce * t_volume).sum() / t_vol_oce

        return tke_integ_miz, tke_integ_ice, tke_integ_oce

    def calc_KE(self, depth=None):

        # load and slice
        uvel = self.get_and_trim_ds(self.preamble + "grid_U.nc").uo_e3u
        vvel = self.get_and_trim_ds(self.preamble + "grid_V.nc").vo_e3v
        wvel = self.get_and_trim_ds(self.preamble + "grid_W.nc").wo_e3w
        e3w = self.get_and_trim_ds(self.preamble + "grid_W.nc").e3w
        e3t = self.get_and_trim_ds(self.preamble + "grid_T.nc").e3t
        rho = self.get_and_trim_ds(self.preamble + "grid_T.nc").rhd
        b_flux = self.get_and_trim_ds(
                  self.preamble + "grid_T.nc").ketrd_convP2K_e3t
        umom = self.get_and_trim_ds(self.preamble + "momu_wm.nc")
        vmom = self.get_and_trim_ds(self.preamble + "momv_wm.nc")
        u_KE = self.get_and_trim_ds(self.preamble + "u_KE_mean.nc")
        v_KE = self.get_and_trim_ds(self.preamble + "v_KE_mean.nc")

        #uvel = xr.open_dataset(self.preamble + 'grid_U.nc', chunks="auto").uo_e3u
        #vvel = xr.open_dataset(self.preamble + 'grid_V.nc', chunks="auto").vo_e3v
        #wvel = xr.open_dataset(self.preamble + 'grid_W.nc', chunks="auto").wo_e3w
        #e3w = xr.open_dataset(self.preamble + 'grid_W.nc', chunks="auto").e3w
        #rho = xr.open_dataset(self.preamble + 'grid_T.nc', chunks="auto").rhd
        #e3t = xr.open_dataset(self.preamble + 'grid_T.nc', chunks="auto").e3t
        #b_flux = xr.open_dataset(self.preamble + 'grid_T.nc', chunks="auto").ketrd_convP2K_e3t
        #umom = xr.open_dataset(self.preamble + 'momu_wm.nc', chunks="auto")
        #vmom = xr.open_dataset(self.preamble + 'momv_wm.nc', chunks="auto")
        #u_KE = xr.open_dataset(self.preamble + 'u_KE_mean.nc', chunks="auto")
        #v_KE = xr.open_dataset(self.preamble + 'v_KE_mean.nc', chunks="auto")

        # get b_bar * w_bar
        b_flux_mean = xr.DataArray(self.wke(rho, wvel, e3w, e3t),
                                   dims=("time_counter", "deptht", "y", "x"),
                                   coords=rho.coords)

        #uvel = uvel.sel(depthu=depth, method='nearest')
        #vvel = vvel.sel(depthv=depth, method='nearest')
        #umom = umom.sel(depthu=depth, method='nearest')
        #vmom = vmom.sel(depthv=depth, method='nearest')
        #u_KE = u_KE.sel(depthu=depth, method='nearest')
        #v_KE = v_KE.sel(depthv=depth, method='nearest')

        if depth: # slice depth
            uvel = uvel.isel(depthu=depth)
            vvel = vvel.isel(depthv=depth)
            umom = umom.isel(depthu=depth)
            vmom = vmom.isel(depthv=depth)
            u_KE = u_KE.isel(depthu=depth)
            v_KE = v_KE.isel(depthv=depth)
            b_flux_mean = b_flux_mean.isel(deptht=depth)
            b_flux = b_flux.isel(deptht=depth)
            e3t = e3t.isel(deptht=depth)

        #fig, (ax0,ax1,ax2)  =plt.subplots(3)
        #bound=1e-5
        #ax0.pcolor(b_flux_mean.isel(time_counter=0), vmin=-bound, vmax=bound, cmap=plt.cm.RdBu)
        #ax1.pcolor(b_flux.isel(time_counter=0), vmin=-bound, vmax=bound, cmap=plt.cm.RdBu)
        #ax2.pcolor(b_flux_mean.isel(time_counter=0) - b_flux.isel(time_counter=0), vmin=-bound, vmax=bound, cmap=plt.cm.RdBu)
        #plt.show()

        uvar_drop = ["time_counter_bounds", "time_centered_bounds",
                     "depthu_bounds", "bounds_nav_lon", "bounds_nav_lat"]
        vvar_drop = ["time_counter_bounds", "time_centered_bounds",
                     "depthv_bounds", "bounds_nav_lon", "bounds_nav_lat"]
        #uvar_drop = ["time_counter_bounds", "time_instant_bounds",
        #             "depthu_bounds", "bounds_nav_lon", "bounds_nav_lat"]
        #vvar_drop = ["time_counter_bounds", "time_instant_bounds",
        #             "depthv_bounds", "bounds_nav_lon", "bounds_nav_lat"]

        umom = umom.drop(uvar_drop)
        vmom = vmom.drop(vvar_drop)
        u_KE = u_KE.drop(uvar_drop)
        v_KE = v_KE.drop(vvar_drop)

        for var in umom.data_vars:
            umom = umom.rename({var:var.lstrip('u').rstrip('u')})
        for var in vmom.data_vars:
            vmom = vmom.rename({var:var.lstrip('v').rstrip('v')})
        for var in u_KE.data_vars:
            print (var)
            u_KE = u_KE.rename({var:var.lstrip('u').rstrip('u_KE')})
        for var in v_KE.data_vars:
            v_KE = v_KE.rename({var:var.lstrip('v').rstrip('v_KE')})

        # get mean kinetic energy
        uvel = uvel.transpose(*list(umom.dims.keys()))
        vvel = vvel.transpose(*list(vmom.dims.keys()))
        uvel["time_counter"] = umom.time_counter
        vvel["time_counter"] = vmom.time_counter
        #EKE = 0.5 * ((umom * uvel) + (vmom * vvel))

        #cfg  = xr.open_dataset(self.path + 'domain_cfg.nc',
        #                       chunks=-1).squeeze()
        cfg = self.get_and_trim_ds(self.path + "domain_cfg.nc").squeeze()

        bu = cfg.e1u * cfg.e2u
        bv = cfg.e1v * cfg.e2v
        self.bt = cfg.e1t * cfg.e2t * e3t

        # Note: 2d variables are broadcast to 3d
        #EKE = 0.5 * ((u_KE - umom * uvel)**2 + (v_KE - vmom * vvel)**2)
        uke = (u_KE - uvel * umom) * bu
        vke = (v_KE - vvel * vmom) * bv
        

        # coordinate hack
        uke = uke.rename({'depthu':'deptht'})
        vke = vke.rename({'depthv':'deptht'})

        # TODO: needs division by e3t
        EKE = 0.25 * ( uke + self.ip1(uke) + vke + self.jp1(vke) ) / self.bt 

        EKE["b_flux_eke"] = ( b_flux - b_flux_mean ) / e3t

        EKE["trd_hpg_e3"] = EKE.trd_hpg_e3 - EKE.b_flux_eke



        # load for faster plotting
        with ProgressBar():
            EKE.load()
        #EKE = 0.5 * ((u_KE) + (v_KE))

        return EKE

    def wke(self, rho, w, e3w, e3t):
        """ get vertical buoyancy flux """

        g = 9.80665
        rho_0 = 1026
        # Local constant initialization
        zcoef = - rho_0 * g * 0.5


        # initialise array
        zconv = np.zeros_like(rho.data)
        print (zconv.shape)

        w = w.data
        e3w = e3w.data
        e3t = e3t.data
        rho = rho.data

        # Surface value (also valid in partial step case)
        zconv[:,0] = zcoef * ( 2 * rho[:,0] ) * w[:,0]# * e3w[:,0]

        print (zconv[:,1:].shape)
        print (rho[:,1:].shape)
        print (rho[:,:-1].shape)
        print (w[:,1:].shape)
        print (e3w[:,1:].shape)
        # interior value (2=<jk=<jpkm1)
        zconv[:,1:] = zcoef * ( rho[:,1:] + rho[:,:-1] ) * w[:,1:]# * e3w[:,1:]

        zconv[:,:-1] = zconv[:,1:] + zconv[:,:-1]

        # conv value on T-point
        zcoef = 0.5 #/ e3t
        pconv = zcoef *  zconv

        return pconv


    def im1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(x=-1, roll_coords=False)

    def ip1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(x=1, roll_coords=False)

    def jm1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(y=-1, roll_coords=False)

    def jp1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(y=1, roll_coords=False)

    def km1(self, var, dvar='deptht'):
        ''' rolling opperations: roll down '''

        return var.roll({dvar:-1}, roll_coords=False)

    def kp1(self, var, dvar='deptht'):
        ''' rolling opperations: roll up '''

        return var.roll({dvar:1}, roll_coords=False)


    def plot_KE(self):

        EKE = self.calc_KE(depth=None).isel(time_counter=0)

        self.depth_inegral(EKE)

        fig, axs = plt.subplots(2,5, figsize=(6.5,4))

        var_bounds = []
        for var in EKE.data_vars:
            if var in ["area"]: continue
            var_bounds.append(abs(EKE[var]).quantile(0.65))
        v_bound = max(var_bounds)
        vmin, vmax = -v_bound, v_bound
        cmap = plt.cm.RdBu_r

        axs[0,0].pcolor(EKE.trd_hpg_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,1].pcolor(EKE.trd_keg_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,2].pcolor(EKE.trd_pvo_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,3].pcolor(EKE.trd_tfr_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0,4].pcolor(EKE.trd_rvo_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[0,4].pcolor(EKE.trd_tau_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,0].pcolor(EKE.trd_zdf_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,1].pcolor(EKE.trd_zad_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,2].pcolor(EKE.trd_bfr_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,3].pcolor(EKE.trd_tot_e3, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1,4].pcolor(EKE.b_flux_eke, vmin=vmin, vmax=vmax, cmap=cmap)

        #axs[0,0].pcolor(EKE.trd_hpg, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[0,1].pcolor(EKE.trd_keg, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[0,2].pcolor(EKE.trd_pvo, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[0,3].pcolor(EKE.trd_tfr, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[0,4].pcolor(EKE.trd_rvo, vmin=vmin, vmax=vmax, cmap=cmap)
        ##axs[0,4].pcolor(EKE.trd_tau, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[1,0].pcolor(EKE.trd_zdf, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[1,1].pcolor(EKE.trd_zad, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[1,2].pcolor(EKE.trd_bfr, vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[1,3].pcolor(EKE.trd_tot, vmin=vmin, vmax=vmax, cmap=cmap)

        # titles
        titles = ['Horiz. Pressure\nGradient',
                  'Advection',
                  'Coriolis',
                  'Ice-Ocean Drag',
                  'Barotropic\nInstability',
                  'Vertical Diffusion',
                  'Vertical Adv',
                  'bottom friction',
                  'Tendency',
                  'Baroclinic\nInstability' ]

        for i, ax in enumerate(axs.flatten()):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal')
        plt.show()


    def plot_EKE_domain_integral_partitioned(self):
        """
        plot domain integrated EKE
        """

        EKE = self.calc_KE(depth=None).isel(time_counter=0)
        EKE = self.mask_mld(EKE)
        EKE = self.depth_integral(EKE)
        EKE_miz, EKE_ice, EKE_oce = self.partition_by_ice_cover(EKE)

        # ini figure
        fig, axs = plt.subplots(1, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.13, right=0.95, top=0.98, bottom=0.19)

        # set list of terms
        var_list = [
        'trd_hpg_e3',
        'trd_keg_e3',
        'trd_pvo_e3',
        'trd_tfr_e3',
        'trd_rvo_e3',
        'trd_zdf_e3',
        'trd_zad_e3',
        'trd_bfr_e3',
        'trd_tot_e3',
        'b_flux_eke',
        ]

        # titles
        titles = ['Horiz. Pressure\nGradient',
                  'Advection',
                  'Coriolis',
                  'Ice-Ocean Drag',
                  'Barotropic\nInstability',
                  'Vertical Diffusion',
                  'Vertical Adv',
                  'bottom friction',
                  'Tendency',
                  'Baroclinic\nInstability' ]

        # render data
        x = np.arange(len(var_list))
        width = 0.25

        # render miz
        data_miz = [EKE_miz[var].values for var in var_list]
        axs.bar(x, data_miz, width, label='MIZ')

        # render ice
        data_ice = [EKE_ice[var].values for var in var_list]
        axs.bar(x + width, data_ice, width, label='Ice')

        # render oce
        data_oce = [EKE_oce[var].values for var in var_list]
        axs.bar(x + width * 2, data_oce, width, label='Oce')

        # legend
        axs.legend(bbox_to_anchor=[1.01,1])

        # set tickes
        axs.set_xticks(x + width, titles)

        # set axis labels
        axs.set_xlabel('Component')
        axs.set_ylabel(r'EKE (m$^2$s$^{-3})$')

        plt.show()

    def plot_EKE_domain_integral(self):
        """
        plot domain integrated EKE
        """

        EKE = self.calc_KE(depth=None).isel(time_counter=0)
        EKE = self.mask_mld(EKE)
        EKE_int = self.domain_integral(EKE)

        # ini figure
        fig, axs = plt.subplots(1, figsize=(6.5,3.5))
        plt.subplots_adjust(left=0.13, right=0.95, top=0.98, bottom=0.19)

        # set list of terms
        var_list = [
        'trd_hpg_e3',
        'trd_keg_e3',
        'trd_pvo_e3',
        'trd_tfr_e3',
        'trd_rvo_e3',
        'trd_zdf_e3',
        'trd_zad_e3',
        'trd_bfr_e3',
        'trd_tot_e3',
        'b_flux_eke',
        ]

        # titles
        titles = ['Horiz. Pressure\nGradient',
                  'Advection',
                  'Coriolis',
                  'Ice-Ocean Drag',
                  'Barotropic\nInstability',
                  'Vertical Diffusion',
                  'Vertical Adv',
                  'bottom friction',
                  'Tendency',
                  'Baroclinic\nInstability' ]

        # render data
        data = [EKE_int[var].values for var in var_list]
        axs.bar(titles, data)

        plt.show()

    def plot_EKE_residual(self):

        EKE = self.calc_KE().isel(time_counter=0)

        fig, axs = plt.subplots(3, figsize=(6.5,4))

        vmin, vmax = -1e-13, 1e-13
        cmap = plt.cm.RdBu_r

        # sum
        trd_RHS = EKE.trd_hpg_e3 + EKE.trd_keg_e3 + \
                  +EKE.trd_pvo_e3 + EKE.trd_rvo_e3 +  \
                  EKE.trd_zdf_e3 + EKE.trd_zad_e3 
                  #EKE.trd_bfr_e3 
        trd_resid = EKE.trd_tot_e3 - trd_RHS

        # plot
        axs[0].pcolor(trd_resid, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].pcolor(trd_RHS, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].pcolor(EKE.trd_tot_e3, vmin=vmin, vmax=vmax, cmap=cmap)

        axs[0].text(0.5, 1.01, "residual", va="bottom", ha="center",
                    transform=axs[0].transAxes, fontsize=8)
        axs[1].text(0.5, 1.01, "RHS", va="bottom", ha="center",
                    transform=axs[1].transAxes, fontsize=8)
        axs[2].text(0.5, 1.01, "total", va="bottom", ha="center",
                    transform=axs[2].transAxes, fontsize=8)

        plt.show()



    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
#file_id = 'SOCHIC_PATCH_15mi_20121209_20121211_'
file_id = 'SOCHIC_PATCH_1d_20121223_20121224_'
ke = plot_KE_Tedesco('EXP02', file_id)
#ke.plot_domain_integrated_TKE_budget_ice_oce_zones()
#ke.plot_laterally_integrated_TKE_budget_ice_oce_zones()
#ke.plot_ke_time_series()
#ke.plot_ml_integrated_TKE_budget()
#print ('depth integrated - done')
#ke.plot_z_slice_KE(depth=10)
#ke.plot_EKE_residual()
#ke.plot_KE()
#ke.plot_EKE_domain_integral()
ke.plot_EKE_domain_integral_partitioned()
print ('depth slice - done')
#ke.plot_domain_integrated_TKE_budget()
