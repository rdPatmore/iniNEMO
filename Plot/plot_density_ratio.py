import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import cmocean

class plot_buoyancy_ratio(object):

    def __init__(self, case, subset=''):

        self.case = case
        self.subset = subset
        if subset == '':
            self.subset_var = ''
        else:
            self.subset_var = '_' + subset

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'

    #def save_10_m_vars(self):
    def load_basics(self):

        chunks = {'time_counter':1,'deptht':1}
        self.bg = xr.open_dataset(config.data_path() + self.case +
                             self.file_id + 'bg.nc',
                             chunks=chunks)
        self.T = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'grid_T.nc',
                              chunks=chunks).votemper
        self.S = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'grid_T.nc',
                              chunks=chunks).vosaline
        self.alpha = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'alpha.nc',
                              chunks=chunks).to_array().squeeze()
        self.beta = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'beta.nc',
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

        # assign index for x and y for merging
        self.bg = self.bg.assign_coords({'x':np.arange(1,self.bg.sizes['x']+1),
                                         'y':np.arange(1,self.bg.sizes['y']+1)})
        self.T = self.T.assign_coords(
                                        {'x':np.arange(self.T.sizes['x']),
                                         'y':np.arange(self.T.sizes['y'])})
        self.S = self.S.assign_coords(
                                        {'x':np.arange(self.S.sizes['x']),
                                         'y':np.arange(self.S.sizes['y'])})
        self.alpha = self.alpha.assign_coords(
                                        {'x':np.arange(self.alpha.sizes['x']),
                                         'y':np.arange(self.alpha.sizes['y'])})
        self.beta = self.beta.assign_coords(
                                        {'x':np.arange(self.beta.sizes['x']),
                                         'y':np.arange(self.beta.sizes['y'])})


        #self.bg.to_netcdf(config.data_path() + self.case + '/TenMetreVars' + 
        #             self.file_id + 'bg_10.nc')
        #self.T.to_netcdf(config.data_path() + self.case + '/TenMetreVars' + 
        #             self.file_id + 'T_10.nc')
        #self.S.to_netcdf(config.data_path() + self.case + '/TenMetreVars' + 
        ##             self.file_id + 'S_10.nc')
        #self.alpha.to_netcdf(config.data_path() + self.case + '/TenMetreVars' + 
        #             self.file_id + 'alpha_10.nc')
        #self.beta.to_netcdf(config.data_path() + self.case + '/TenMetreVars' + 
        #             self.file_id + 'beta_10.nc')
       # 

   #     self.bg = xr.open_dataset(config.data_path() + self.case + 
   #                          '/TenMetreVars' + self.file_id + 'bg_10.nc',
   #                           chunks='auto')
   #     self.T = xr.open_dataset(config.data_path() + self.case + 
   #                          '/TenMetreVars' + self.file_id + 'T_10.nc',
   #                           chunks='auto')
   #     self.S = xr.open_dataset(config.data_path() + self.case + 
   #                          '/TenMetreVars' + self.file_id + 'S_10.nc',
   #                           chunks='auto')
   #     self.alpha = xr.open_dataset(config.data_path() + self.case + 
   #                          '/TenMetreVars' + self.file_id + 'alpha_10.nc',
   #                           chunks='auto')
   #     self.beta = xr.open_dataset(config.data_path() + self.case + 
   #                          '/TenMetreVars' + self.file_id + 'beta_10.nc',
   #                           chunks='auto')

        self.cfg = xr.open_dataset(config.data_path() + self.case +
                                   '/domain_cfg.nc').squeeze()

        # assign index for x and y for merging
        self.cfg = self.cfg.assign_coords(
                                        {'x':np.arange(self.cfg.sizes['x']),
                                         'y':np.arange(self.cfg.sizes['y'])})

        # add norm
        self.bg['norm_grad_b'] = (self.bg.dbdx**2 + self.bg.dbdy**2)**0.5

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
            self.T = self.T.where(self.T.nav_lat>-59.9858036,
                                          drop=True)
            self.S = self.T.where(self.S.nav_lat>-59.9858036,
                                          drop=True)
            self.alpha = self.alpha.where(self.alpha.nav_lat>-59.9858036,
                                          drop=True)
            self.beta = self.beta.where(self.beta.nav_lat>-59.9858036,
                                          drop=True)
            self.cfg = self.cfg.where(self.cfg.gphit>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)
            self.T = self.T.where(self.T.nav_lat<-59.9858036, 
                                          drop=True)
            self.S = self.S.where(self.S.nav_lat<-59.9858036, 
                                          drop=True)
            self.alpha = self.alpha.where(self.alpha.nav_lat<-59.9858036, 
                                          drop=True)
            self.beta = self.beta.where(self.beta.nav_lat<-59.9858036, 
                                          drop=True)
            self.cfg = self.cfg.where(self.cfg.gphit<-59.9858036, drop=True)

        self.giddy_raw = xr.open_dataset(config.root() +
                                         'Giddy_2020/merged_raw.nc')

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

    def get_grad_T_and_S(self, load=False, save=False, giddy_method=False):
        ''' get dCdx and dCdy where C=[T,S] '''

        if load:
            self.TS_grad = xr.open_dataset(config.data_path() + self.case +
                                       self.file_id + 'TS_grad_10m' +
                                       self.subset_var + '.nc')
        else:
            dx = self.cfg.e1t
            dy = self.cfg.e2t
            if giddy_method: 
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
        dr_x = np.abs(self.TS_grad.alpha_dTdx / self.TS_grad.beta_dSdx)
        dr_y = np.abs(self.TS_grad.alpha_dTdy / self.TS_grad.beta_dSdy)
        dr   = np.abs(self.TS_grad.gradTrho / self.TS_grad.gradSrho)

        # drop inf
        dr_x = xr.where(np.isinf(dr_x), np.nan, dr_x)
        dr_y = xr.where(np.isinf(dr_y), np.nan, dr_y)
        dr   = xr.where(np.isinf(dr  ), np.nan, dr  )

        dr_x.name = 'density_ratio_x'
        dr_y.name = 'density_ratio_y'
        dr.name   = 'density_ratio_norm'
        
        self.density_ratio = xr.merge([dr_x, dr_y, dr])
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
   
        if load:
            self.stats = xr.open_dataset(config.data_path() + self.case + 
                self.file_id + 'density_ratio_stats_giddy_method'
                                      + self.subset_var + '.nc')
        else:
            self.get_grad_T_and_S()
            self.get_density_ratio()

            self.TS_grad = self.get_stats(self.TS_grad)
            self.bg = self.get_stats(self.bg)
            self.density_ratio = self.get_stats(self.density_ratio)

            self.stats = xr.merge([self.TS_grad, self.bg, self.density_ratio])

            # save
            if save:
                self.stats.to_netcdf(config.data_path() + self.case + 
                self.file_id + 'density_ratio_stats_giddy_method'
                                      + self.subset_var + '.nc')
                                       #self.file_id + 'density_ratio_stats.nc')

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

    def render_Ro_sea_ice(self, fig,  axs):
        ### this needs to be transformed to albers projection!
    
        # load sea ice and Ro
        si = xr.open_dataset(config.data_path() + 
                     'EXP10/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc').siconc
        Ro = xr.open_dataset(config.data_path_old() + 
                        'EXP10/rossby_number.nc').Ro
        # plot sea ice
        halo=(1%3) + 1
        si = si.sel(time_counter='2012-12-30 00:00:00', method='nearest')
        si = si.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        
        p0 = axs[0].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
                           cmap=cmocean.cm.ice, vmin=0, vmax=1)
        axs[0].set_aspect('equal')
        axs[0].set_xlabel('Longitude')
        axs[0].set_xlim([-3.7,3.7])
        axs[0].set_ylim([-63.8,-56.2])
    
        # plot Ro
        Ro = Ro.sel(time_counter='2012-12-30 00:00:00', method='nearest')
        Ro = Ro.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        Ro = Ro.isel(depth=10)
        
        p1 = axs[1].pcolor(Ro.nav_lon, Ro.nav_lat, Ro, shading='nearest',
                      cmap=plt.cm.RdBu, vmin=-0.45, vmax=0.45)
        axs[1].set_aspect('equal')
        axs[1].set_xlabel('Longitude')
        axs[1].set_xlim([-3.7,3.7])
        axs[1].set_ylim([-63.8,-56.2])
    
        axs[1].yaxis.set_ticklabels([])
        
        axs[0].set_ylabel('Latitude')
        
        pos = axs[0].get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -4.3, r'Sea Ice Concentration', fontsize=8,
                rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')
    
        pos = axs[1].get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -4.3, r'$\zeta / f$', fontsize=8,
                rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')

        axs[0].text(0.1, 0.9, '2012-12-30', transform=axs[0].transAxes, c='w')

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
        fig = plt.figure(figsize=(5.5, 4.5), dpi=300)
        self.get_T_S_bg_stats(load=True)

        #rho seaice
        #plt.subplots_adjust(bottom=0.3, top=0.99, right=0.99, left=0.1,
        #                    wspace=0.05)

        gs0 = gridspec.GridSpec(ncols=1, nrows=4)
        gs1 = gridspec.GridSpec(ncols=2, nrows=1)
        gs0.update(top=0.99, bottom=0.55, left=0.1, right=0.98, hspace=0.05)
        gs1.update(top=0.50, bottom=0.1,  left=0.1, right=0.98, wspace=0.05)

        axs0, axs1 = [], []
        for i in range(4):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(2):
            axs1.append(fig.add_subplot(gs1[i]))

        # tan of density ratio
        self.stats['turner_angle_norm_ts_mean'] = np.arctan(
                                       self.stats.density_ratio_x_ts_mean)/np.pi
        self.stats['turner_angle_norm_ts_std'] = np.arctan(
                                       self.stats.density_ratio_y_ts_std)/np.pi

        def render(ax, var):
            ax.plot(self.stats.time_counter, self.stats[var])

        def render_density_ratio(ax, var):
            var_mean = var + '_ts_mean'
            lower = self.stats[var_mean].where(self.stats[var_mean] < 0.25)
            upper = self.stats[var_mean].where(self.stats[var_mean] > 0.25)
            ax.fill_between(lower.time_counter, lower, 0.25,
                            edgecolor=None, color='teal')
            ax.fill_between(upper.time_counter, 0.25, upper,
                            edgecolor=None, color='tab:red')

        # render Temperature contirbution
        render(axs0[0], 'gradTrho_ts_mean')
        #render(axs0[0], 'gradTrho_ts_mean_north')
        #render(axs0[0], 'gradTrho_ts_mean_south')
        render(axs0[1], 'gradTrho_ts_std')
        #render(axs0[1], 'gradTrho_ts_std_north')
        #render(axs0[1], 'gradTrho_ts_std_south')
        #render(axs0[2], 'gradT_ts_std') # qt_ocean and sfx
        render_density_ratio(axs0[3], 'density_ratio_norm')

        self.render_Ro_sea_ice(fig, axs1)

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

#for subset in ['north', 'south', '']:
#    m = plot_buoyancy_ratio('EXP10', subset=subset)
#    m.load_basics()
#    m.get_grad_T_and_S(save=True)
#    m.get_density_ratio()
#    m.get_T_S_bg_stats(save=True)

               ### plot ###
m = plot_buoyancy_ratio('EXP10')
#m.get_T_S_bg_stats(load=True)
m.plot_density_ratio_with_SI_Ro_and_bg_time_series()
