import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import cmocean
import cartopy.crs as ccrs

class plot_buoyancy_ratio(object):

    def __init__(self, case, subset='', giddy_method=False):

        self.case = case
        self.subset = subset
        self.giddy_method = giddy_method
        if subset == '':
            self.subset_var = ''
        else:
            self.subset_var = '_' + subset

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.f_path  = config.data_path() + case + self.file_id 

    def subset_n_s(self, arr, loc='north'):
        if loc == 'north':
            arr = arr.where(arr.nav_lat>-59.9858036, drop=True)
        if loc == 'south':
            arr = arr.where(arr.nav_lat<-59.9858036, drop=True)
        return arr

    def assign_x_y_index(self.arr, shift=0)
        arr = arr.assign_coords({'x':np.arange(shift,arr.sizes['x']+shift),
                                 'y':np.arange(shift,arr.sizes['y']+shift)})
        return arr

    def load_basics(self, surface_fluxes=False):

        chunks = {'time_counter':1,'deptht':1}
        self.bg = xr.open_dataset(self.f_path + 'bg.nc', chunks=chunks)
        self.T = xr.open_dataset(self.f_path + 'grid_T.nc', 
                                 chunks=chunks).votemper
        self.S = xr.open_dataset(self.f_path + 'grid_T.nc',
                                 chunks=chunks).vosaline
        self.alpha = xr.open_dataset(self.f_path + 'alpha.nc',
                                     chunks=chunks).to_array().squeeze()
        self.beta = xr.open_dataset(self.f_path + 'beta.nc',
                                    chunks=chunks).to_array().squeeze()

        # name the arrays (this should really be done in model_object
        #                  where the data is made)
        self.alpha.name = 'alpha'
        self.beta.name = 'beta'

        # make bg nameing consistent
        self.bg = self.bg.rename({'bx':'dbdx', 'by':'dbdy'})

        # restrict to 10 m
        sel_kwargs = dict(deptht=10, method='nearest')
        self.bg = self.bg.sel(**sel_kwargs)#.load()
        self.T = self.T.sel(**sel_kwargs)#.load()
        self.S = self.S.sel(**sel_kwargs)#.load()
        self.alpha = self.alpha.sel(**sel_kwargs)#.load()
        self.beta = self.beta.sel(**sel_kwargs)#.load()

        # load grid
        self.cfg = xr.open_dataset(config.data_path() + self.case +
                                   '/domain_cfg.nc').squeeze()

        # assign index for x and y for merging
        self.bg    = self.assign_x_y_index(self.bg, shift=1)
        self.T     = self.assign_x_y_index(self.T)
        self.S     = self.assign_x_y_index(self.S)
        self.alpha = self.assign_x_y_index(self.alpha)
        self.beta  = self.assign_x_y_index(self.beta)
        self.cfg   = self.assign_x_y_index(self.cfg)

        # add norm
        self.bg['norm_grad_b'] = (self.bg.dbdx**2 + self.bg.dbdy**2)**0.5

        # subset model
        if self.subset:
            self.bg     = self.subset_n_s(self.bg,     loc=self.subset) 
            self.T      = self.subset_n_s(self.T,      loc=self.subset) 
            self.S      = self.subset_n_s(self.S,      loc=self.subset) 
            self.alpha  = self.subset_n_s(self.alpha,  loc=self.subset) 
            self.beta   = self.subset_n_s(self.beta,   loc=self.subset) 
            self.cfg    = self.subset_n_s(self.cfg,    loc=self.subset) 

        self.giddy_raw = xr.open_dataset(config.root() +
                                         'Giddy_2020/merged_raw.nc')

    def load_surface_fluxes(self):
        '''
        load
            - sfx   : surface freshwater fluxes
            - qt_oce: surface heat fluxes
        '''

        # load
        chunks = {'time_counter':1,'deptht':1}
        self.sfx    = xr.open_dataset(self.f_path + 'grid_T.nc',
                                   chunks=chunks).sfx
        self.qt_oce = xr.open_dataset(self.f_path + 'grid_T.nc',
                                      chunks=chunks).qt_oce

        # assign index for x and y for merging
        self.sfx    = self.assign_x_y_index(self.sfx)
        self.qt_oce = self.assign_x_y_index(self.qt_oce)

        # subset model
        if self.subset:
            self.sfx    = self.subset_n_s(self.sfx,    loc=self.subset) 
            self.qt_oce = self.subset_n_s(self.qt_oce, loc=self.subset) 


    def restrict_to_glider_time(self, ds):
        '''
        restrict the model time to glider time 
        raw glider does not work for this
        currently not functional
        '''
    
        clean_float_time = self.giddy_raw.ctd_time
        start = clean_float_time.min().astype('datetime64[ns]').values
        end   = clean_float_time.max().astype('datetime64[ns]').values

        ds = ds.sel(time_counter=slice(start,end))
        return ds

    def get_grad_T_and_S(self, load=False, save=False):
        ''' get dCdx and dCdy where C=[T,S] '''

        if load:
            self.TS_grad = xr.open_dataset(config.data_path() + self.case +
                                       self.file_id + 'TS_grad_10m' +
                                       self.subset_var + '.nc')
        else:
            dx = self.cfg.e1t
            dy = self.cfg.e2t
            if self.giddy_method: 
                # take scalar gradients only
                dTx = self.T.diff('x').pad(x=(1,0), constant_value=0) # u-pts
                dTy = self.T.diff('y').pad(y=(1,0), constant_value=0) # v-pts
                dSx = self.S.diff('x').pad(x=(1,0), constant_value=0) # u-pts
                dSy = self.S.diff('y').pad(y=(1,0), constant_value=0) # v-pts
                
                dTdx = self.alpha * dTx / dx
                dTdy = self.alpha * dTy / dy
                dSdx = self.beta  * dSx / dx
                dSdy = self.beta  * dSy / dy
            else:
                # gradient of alpha and beta included
                rhoT = self.alpha * self.T
                rhoS = self.beta  * self.S

                dTdx = rhoT.diff('x').pad(x=(1,0), constant_value=0) # u-pts
                dTdy = rhoT.diff('y').pad(x=(1,0), constant_value=0) # v-pts
                dSdx = rhoS.diff('x').pad(x=(1,0), constant_value=0) # u-pts
                dSdy = rhoS.diff('y').pad(x=(1,0), constant_value=0) # v-pts

            # get norms
            gradT = (dTdx**2 + dTdy**2)**0.5
            gradS = (dSdx**2 + dSdy**2)**0.5
           
            # name
            dTdx.name = 'alpha_dTdx'
            dTdy.name = 'alpha_dTdy'
            dSdx.name = 'beta_dSdx'
            dSdy.name = 'beta_dSdy'
            gradT.name = 'gradTrho'
            gradS.name = 'gradSrho'
            
            self.TS_grad = xr.merge([dTdx,dTdy,dSdx,dSdy,gradT,gradS])
            
            # save
            if save:
                self.TS_grad.to_netcdf(config.data_path() + self.case + 
                                       self.file_id + 'TS_grad_10m' +
                                       self.subset_var + '.nc')

    def get_density_ratio(self):
        ''' 
        get  (alpha * dTdx)/ (beta * dSdx)
             (alpha * dTdy)/ (beta * dSdy)
        '''

        # nan/inf issues are with TS_grad
        # density ratio vectors
        dr_x = np.abs(self.TS_grad.alpha_dTdx / self.TS_grad.beta_dSdx)
        dr_y = np.abs(self.TS_grad.alpha_dTdy / self.TS_grad.beta_dSdy)

        # 2d denstiy ratio, where 1 is dividing line between T and S dominance
        # see Ferrari and Paparella ((2004)
        Tu_comp = (self.TS_grad.alpha_dTdx + 1j * self.TS_grad.alpha_dTdy)  \
                / (self.TS_grad.beta_dSdx  + 1j * self.TS_grad.beta_dSdy )

        # get complex argument (angle
        # workaround :: np.angle is yet to be dask enabled
        def get_complex_arg(arr):
            return np.angle(arr)
        Tu_phi = xr.apply_ufunc(get_complex_arg, Tu_comp, dask='parallelized')

        # get the complex modulus
        Tu_mod = np.abs(Tu_comp)

        # drop inf
        dr_x = xr.where(np.isinf(dr_x), np.nan, dr_x)
        dr_y = xr.where(np.isinf(dr_y), np.nan, dr_y)

        dr_x.name = 'density_ratio_x'
        dr_y.name = 'density_ratio_y'
        Tu_phi.name   = 'density_ratio_2d_angle'
        Tu_mod.name   = 'density_ratio_2d_mod'
        
        self.density_ratio = xr.merge([dr_x, dr_y, Tu_phi, Tu_mod])
        print ('merged')

    def get_stats(self, ds):
        ''' 
           get model mean and std time_series
               - buoyancy
               - theta
               - salinity
        '''

        ds_mean  = np.abs(ds).mean(['x','y'], skipna=True)
        ds_std   = np.abs(ds).std(['x','y'], skipna=True)
        chunks = dict(x=-1, y=-1)
        ds_quant   = np.abs(ds.chunk(chunks)).quantile([0.05, 0.95],
                     ['x','y'], skipna=True)
        for key in ds.keys():
            ds_mean  = ds_mean.rename({key: key + '_ts_mean' + self.subset_var})
            ds_std  = ds_std.rename({key: key + '_ts_std'+ self.subset_var})
            ds_quant  = ds_quant.rename(
                                     {key: key + '_ts_quant' + self.subset_var})

        ds_stats = xr.merge([ds_mean, ds_std, ds_quant]).load()
 
        return ds_stats

    def get_T_S_bg_stats(self, save=False, load=False):
        '''
        get mean, std and quantiles for
            - bg
            - density ratio
            - T/S gradient
        '''

        # define save str for giddy method   
        if self.giddy_method:
            giddy_str = '_giddy_method'
        else:
            giddy_str = ''

        # define file name
        f = self.f_path + 'density_ratio_stats' + giddy_str + self.subset_var\
            + '.nc'

        if load:
            self.stats = xr.open_dataset(f)

        else:
            self.get_grad_T_and_S()
            self.get_density_ratio()

            self.TS_grad = self.get_stats(self.TS_grad)
            self.bg = self.get_stats(self.bg)
            self.density_ratio = self.get_stats(self.density_ratio)

            self.stats = xr.merge([self.TS_grad, self.bg, self.density_ratio])

            # save
            if save:
                self.stats.to_netcdf(f)

    def get_bg_and_surface_flux_stats(self, save=False, load=False):
        '''
        get mean, std and quantiles for
            - bg
            - density ratio
            - T/S gradient
        '''

        # define file name
        f = self.f_path + 'bg_and_suface_flux_stats' + self.subset_var + '.nc'

        if load:
            self.stats = xr.open_dataset(f) 

        else:
            # get stats (require load_basics and load_surface_fluxes)
            bg_stats = self.get_stats(self.bg)
            sfx_stats = self.get_stats(self.sfx)
            qt_oce_stats = self.get_stats(self.qt_oce)

            # save
            if save:
                self.stats.to_netcdf(f)

    def plot_density_ratio(self):
        '''
        plot - buoyancy gradient
             - temperature gradient
             - salinity gradient
             - density ratio
        over time

        4 x 2 plot with columns of x and y components
        '''

        fig, axs = plt.subplots(4,2, figsize=(5.5,5.5))

        # tan of density ratio
        self.stats['density_ratio_x_ts_mean'] = np.arctan(
                                       self.stats.density_ratio_x_ts_mean)/np.pi
        self.stats['density_ratio_y_ts_mean'] = np.arctan(
                                       self.stats.density_ratio_y_ts_mean)/np.pi
        self.stats['density_ratio_x_ts_std'] = np.arctan(
                                       self.stats.density_ratio_x_ts_std)/np.pi
        self.stats['density_ratio_y_ts_std'] = np.arctan(
                                       self.stats.density_ratio_y_ts_std)/np.pi

        def render(ax, var):
            var_mean = var + '_ts_mean'
            var_std = var + '_ts_std'
            gfac = 1
            c='k'
            if var in ['alpha_dTdx','alpha_dTdy','beta_dSdx','beta_dSdy']:
                gfac = 9.81
                c='green'
            #ax.fill_between(self.stats.time_counter,
            #                    self.stats[var_mean] - self.stats[var_std],
            #                    self.stats[var_mean] + self.stats[var_std],
            #                    edgecolor=None)
            ax.plot(self.stats.time_counter, gfac * self.stats[var_mean], c=c)

        var_list = ['dbdx','dbdy','alpha_dTdx','alpha_dTdy'
                   ,'beta_dSdx','beta_dSdy',
                    'density_ratio_x','density_ratio_y']


        for j, col in enumerate(axs):
            for i, ax in enumerate(col):
                print (var_list[i+(2*j)], i, j)
                render(ax,var_list[i+(2*j)])
        render(axs[0,0], 'alpha_dTdx')
        render(axs[0,0], 'beta_dSdx')
        print (self.stats.density_ratio_x_ts_mean.min())
        print (self.stats.density_ratio_x_ts_mean.max())
        print (self.stats.density_ratio_x_ts_std.min())
        print (self.stats.density_ratio_x_ts_std.max())

        db_lims = (1e-9,5e-8)
        dT_lims = (1e-10,5e-10)
        dS_lims = (3.4e-8,6.5e-8)
        ratio_lims = (0,1.55)
        
        for i in [0,1]:
            #axs[0,i].set_ylim(db_lims)
            #axs[1,i].set_ylim(dT_lims)
            #axs[2,i].set_ylim(dS_lims)
            axs[3,i].set_ylim(ratio_lims)
            axs[3,i].yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
            axs[3,i].yaxis.set_major_locator(
                                   matplotlib.ticker.MultipleLocator(base=0.25))
        plt.savefig('density_ratio.png')

    def plot_density_ratio_two_panel(self):
        '''
        plot - buoyancy gradient
             - density ratio
        over time

        2 x 2 plot with columns of x and y components
        '''

        fig, axs = plt.subplots(2,2, figsize=(5.5,4.0))
        plt.subplots_adjust(top=0.95, right=0.98, left=0.15, bottom=0.2,
                            hspace=0.05, wspace=0.05)

        # tan of density ratio
        self.stats['density_ratio_x_ts_mean'] = np.arctan(
                                       self.stats.density_ratio_x_ts_mean)/np.pi
        self.stats['density_ratio_y_ts_mean'] = np.arctan(
                                       self.stats.density_ratio_y_ts_mean)/np.pi
        self.stats['density_ratio_x_ts_std'] = np.arctan(
                                       self.stats.density_ratio_x_ts_std)/np.pi
        self.stats['density_ratio_y_ts_std'] = np.arctan(
                                       self.stats.density_ratio_y_ts_std)/np.pi

        def render(ax, var):
            var_mean = var + '_ts_mean'
            var_std = var + '_ts_std'
            gfac = 1
            c='k'
            #ax.fill_between(self.stats.time_counter,
            #                    self.stats[var_mean] - self.stats[var_std],
            #                    self.stats[var_mean] + self.stats[var_std],
            #                    edgecolor=None)
            ax.plot(self.stats.time_counter, gfac * self.stats[var_mean], c=c)

        var_list = ['dbdx','dbdy',
                    'density_ratio_x','density_ratio_y']

        def render_density_ratio(ax, var):
            var_mean = var + '_ts_mean'
            lower = self.stats[var_mean].where(self.stats[var_mean] < 0.25)
            upper = self.stats[var_mean].where(self.stats[var_mean] > 0.25)
            ax.fill_between(lower.time_counter, lower, 0.25,
                            edgecolor=None, color='teal')
            ax.fill_between(upper.time_counter, 0.25, upper,
                            edgecolor=None, color='tab:red')


        for j, col in enumerate(axs):
            for i, ax in enumerate(col):
                print (var_list[i+(2*j)], i, j)
                render(ax,var_list[i+(2*j)])
        render_density_ratio(axs[1,0], var_list[2])
        render_density_ratio(axs[1,1], var_list[3])
        print (self.stats.density_ratio_x_ts_mean.min())
        print (self.stats.density_ratio_x_ts_mean.max())
        print (self.stats.density_ratio_x_ts_std.min())
        print (self.stats.density_ratio_x_ts_std.max())

        db_lims = (0,3.6e-8)
        ratio_lims = (0,0.35)
        print (self.stats)
        date_lims = (self.stats.time_counter.min(), 
                     self.stats.time_counter.max())
        
        for i in [0,1]:
            axs[0,i].set_xlim(date_lims)
            axs[1,i].set_xlim(date_lims)
            axs[0,i].set_ylim(db_lims)
            axs[1,i].set_ylim(ratio_lims)
            axs[1,i].yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
            axs[1,i].yaxis.set_major_locator(
                                   matplotlib.ticker.MultipleLocator(base=0.25))
            axs[1,i].axhline(0.25, 0, 1)
            axs[0,i].set_xticklabels([])
            axs[i,1].set_yticklabels([])

            # date labels
            axs[1,i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            # Rotates and right-aligns 
            for label in axs[1,i].get_xticklabels(which='major'):
                label.set(rotation=35, horizontalalignment='right')
            axs[1,i].set_xlabel('date')

        axs[0,0].set_ylabel('buoyancy gradient')
        axs[1,0].set_ylabel('denstiy ratio')

        plt.savefig('density_ratio_two_panel.png')

    def plot_density_ratio_with_SI_Ro_and_bg_time_series(self):
        '''
        plot over time - buoyancy gradient mean (n,s,all)
                       - buoyancy gradient std (n,s,all)
                       - fresh water fluxes and sfx (n,s,all)
                       - density ratio
        plot map - Sea ice concentration
                 - Ro
        GridSpec
         a) 4 x 1
         b) 1 x 2
        '''

        # get stats
        self.get_T_S_bg_stats(load=True)

        fig = plt.figure(figsize=(5.5, 4.5), dpi=300)

        #rho seaice
        #plt.subplots_adjust(bottom=0.3, top=0.99, right=0.99, left=0.1,
        #                    wspace=0.05)

        gs0 = gridspec.GridSpec(ncols=2, nrows=2)
        gs1 = gridspec.GridSpec(ncols=1, nrows=3)
        gs0.update(top=0.99, bottom=0.55, left=0.1, right=0.98, hspace=0.05)
        gs1.update(top=0.50, bottom=0.1,  left=0.1, right=0.98, wspace=0.05)

        axs0, axs1 = [], []
        for i in range(4):
            axs0.append(fig.add_subplot(gs1[i],
                     projection=ccrs.AlbersEqualArea(central_latitude=60,
                      standard_parallels=(-62,-58))))
        for i in range(3):
            axs1.append(fig.add_subplot(gs1[i]))

        def render(ax, var):
            ax.plot(self.stats.time_counter, self.stats[var])

        def render_density_ratio(ax, var):
            var_mean = var + '_ts_mean'
            lower = self.stats[var_mean].where(self.stats[var_mean] < 1)
            upper = self.stats[var_mean].where(self.stats[var_mean] > 1)
            ax.fill_between(lower.time_counter, lower, 1,
                            edgecolor=None, color='teal')
            ax.fill_between(upper.time_counter, 1, upper,
                            edgecolor=None, color='tab:red')

        # render Temperature contirbution
        render(axs0[0], 'gradTrho_ts_mean')
        #render(axs0[2], 'gradT_ts_std') # qt_ocean and sfx
        #render_density_ratio(axs0[3], 'density_ratio_norm')

        # load sea ice and Ro
        si = xr.open_dataset(config.data_path() + 
                     'EXP10/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc').siconc
        Ro = xr.open_dataset(config.data_path_old() + 
                     'EXP10/rossby_number.nc').Ro
        # plot sea ice
        halo=(1%3) + 1 # ???
        si = si.sel(time_counter='2012-12-30 00:00:00', method='nearest')
        si = si.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        
        p0 = axs0[0,0].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
                              cmap=cmocean.cm.ice, vmin=0, vmax=1,
                              projection=ccrs.PlateCarree())
    
        # plot Ro
        Ro = Ro.sel(time_counter='2012-12-30 00:00:00', method='nearest')
        Ro = Ro.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        Ro = Ro.isel(depth=10)
        
        # render
        p1 = axs0[0,1].pcolor(Ro.nav_lon, Ro.nav_lat, Ro, shading='nearest',
                              cmap=plt.cm.RdBu, vmin=-0.45, vmax=0.45,
                              projection=ccrs.PlateCarree())

        # axes formatting
        for ax in axs0[:,0]:
            ax.set_ylabel('Latitude')
        for ax in axs0[1,:]:
            axs[1].set_xlabel('Longitude')
        for ax in axs0[:,1]:
            ax.yaxis.set_ticklabels([])
        for ax in axs0.flatten():
            axs[1].set_xlim([-3.7,3.7])
            axs[1].set_ylim([-63.8,-56.2])
        
        ## colour bars
        #pos = axs[0].get_position()
        #cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
        #cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
        #cbar.ax.text(0.5, -4.3, r'Sea Ice Concentration', fontsize=8,
        #        rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')
    
        #pos = axs[1].get_position()
        #cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
        #cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
        #cbar.ax.text(0.5, -4.3, r'$\zeta / f$', fontsize=8,
        #        rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')

        #axs[0].text(0.1, 0.9, '2012-12-30', transform=axs[0].transAxes, c='w')


        #db_lims = (0,3.6e-8)
        #ratio_lims = (0,0.35)
        #print (self.stats)
        #date_lims = (self.stats.time_counter.min(), 
        #             self.stats.time_counter.max())
        #
        #for i in [0,1]:
        #    axs[0,i].set_xlim(date_lims)
        #    axs[1,i].set_xlim(date_lims)
        #    axs[0,i].set_ylim(db_lims)
        #    axs[1,i].set_ylim(ratio_lims)
        #    axs[1,i].yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
        #    axs[1,i].yaxis.set_major_locator(
        #                           matplotlib.ticker.MultipleLocator(base=0.25))
        #    axs[1,i].axhline(0.25, 0, 1)
        #    axs[0,i].set_xticklabels([])
        #    axs[i,1].set_yticklabels([])

        #    # date labels
        #    axs[1,i].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        #    # Rotates and right-aligns 
        #    for label in axs[1,i].get_xticklabels(which='major'):
        #        label.set(rotation=35, horizontalalignment='right')
        #    axs[1,i].set_xlabel('date')

        #axs[0,0].set_ylabel('buoyancy gradient')
        #axs[1,0].set_ylabel('denstiy ratio')

        plt.savefig('density_ratio_with_SI_Ro_and_bg_time_series.png')


       #### make files ####

# this needs to be run in two rounds, saving intermediate files: 1st/2nd round
for subset in ['north', 'south', '']:
    m = plot_buoyancy_ratio('EXP10', subset=subset)
    m.load_basics()
    #m.get_grad_T_and_S(save=True)  # first round
    m.get_T_S_bg_stats(save=True) # second round

               ### plot ###
#m = plot_buoyancy_ratio('EXP10')
#m.get_T_S_bg_stats(load=True)
#m.plot_density_ratio_with_SI_Ro_and_bg_time_series()
