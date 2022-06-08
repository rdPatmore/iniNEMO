import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib

class plot_buoyancy_ratio(object):

    def __init__(self, case, subset=''):

        self.case = case
        self.subset = subset

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'

    def load_basics(self):
        self.bg = xr.open_dataset(config.data_path() + self.case +
                             self.file_id + 'bg.nc',
                             chunks='auto')
        self.gridT = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'grid_T.nc',
                              chunks='auto')
        self.alpha = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'alpha.nc',
                              chunks='auto').to_array().squeeze()
        self.beta = xr.open_dataset(config.data_path() + self.case +
                              self.file_id + 'beta.nc',
                              chunks='auto').to_array().squeeze()

        # name the arrays (this should really be done in model_object
        #                  where the data is made)
        self.alpha.name = 'alpha'
        self.beta.name = 'beta'

        self.cfg = xr.open_dataset(config.data_path() + self.case +
                                   '/domain_cfg.nc').squeeze()


        # assign index for x and y for merging
        self.bg = self.bg.assign_coords({'x':np.arange(1,self.bg.sizes['x']+1),
                                         'y':np.arange(1,self.bg.sizes['y']+1)})
        self.gridT = self.gridT.assign_coords(
                                        {'x':np.arange(self.gridT.sizes['x']),
                                         'y':np.arange(self.gridT.sizes['y'])})
        self.alpha = self.alpha.assign_coords(
                                        {'x':np.arange(self.alpha.sizes['x']),
                                         'y':np.arange(self.alpha.sizes['y'])})
        self.beta = self.beta.assign_coords(
                                        {'x':np.arange(self.beta.sizes['x']),
                                         'y':np.arange(self.beta.sizes['y'])})
        self.cfg = self.cfg.assign_coords(
                                        {'x':np.arange(self.cfg.sizes['x']),
                                         'y':np.arange(self.cfg.sizes['y'])})

        # make bg nameing consistent
        self.bg = self.bg.rename({'bx':'dbdx', 'by':'dbdy'})

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
            self.gridT = self.gridT.where(self.gridT.nav_lat>-59.9858036,
                                          drop=True)
            self.alpha = self.alpha.where(self.alpha.nav_lat>-59.9858036,
                                          drop=True)
            self.beta = self.beta.where(self.beta.nav_lat>-59.9858036,
                                          drop=True)
            self.cfg = self.cfg.where(self.cfg.gphit>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)
            self.gridT = self.gridT.where(self.gridT.nav_lat<-59.9858036, 
                                          drop=True)
            self.alpha = self.alpha.where(self.alpha.nav_lat<-59.9858036, 
                                          drop=True)
            self.beta = self.beta.where(self.beta.nav_lat<-59.9858036, 
                                          drop=True)
            self.cfg = self.cfg.where(self.cfg.gphit<-59.9858036, drop=True)


        # restrict to 10 m
        sel_kwargs = dict(deptht=10, method='nearest')
        self.bg = self.bg.sel(**sel_kwargs)#.load()
        self.gridT = self.gridT.sel(**sel_kwargs)#.load()
        self.alpha = self.alpha.sel(**sel_kwargs)#.load()
        self.beta = self.beta.sel(**sel_kwargs)#.load()

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
                                    self.file_id +'TS_grad_10m.nc')
        else:
            dx = self.cfg.e1t
            dy = self.cfg.e2t
            if giddy_method: 
                # take scalar gradients only
                dgridTx = self.gridT.diff('x').pad(x=(1,0),
                                          constant_value=0) # u-pts
                dgridTy = self.gridT.diff('y').pad(y=(1,0),
                                          constant_value=0) # v-pts
                
                dTdx = self.alpha * dgridTx.votemper / dx
                dTdy = self.alpha * dgridTy.votemper / dy
                dSdx = self.beta * dgridTx.vosaline / dx
                dSdy = self.beta * dgridTy.vosaline / dy
            else:
                # gradient of alpha and beta included
                rhoT = self.alpha * self.gridT.votemper
                rhoS = self.beta  * self.gridT.vosaline

                dTdx = rhoT.diff('x').pad(x=(1,0), constant_value=0) # u-pts
                dTdy = rhoT.diff('y').pad(x=(1,0), constant_value=0) # v-pts
                dSdx = rhoS.diff('x').pad(x=(1,0), constant_value=0) # u-pts
                dSdy = rhoS.diff('y').pad(x=(1,0), constant_value=0) # v-pts

            # name
            dTdx.name = 'alpha_dTdx'
            dTdy.name = 'alpha_dTdy'
            dSdx.name = 'beta_dSdx'
            dSdy.name = 'beta_dSdy'
            
            self.TS_grad = xr.merge([dTdx,dTdy,dSdx,dSdy])
            
            # save
            if save:
                self.TS_grad.to_netcdf(config.data_path() + self.case + 
                                       self.file_id + 'TS_grad_10m.nc')

    def get_density_ratio(self):
        ''' get
        (alpha * dTdx)/ (beta * dSdx)
        (alpha * dTdy)/ (beta * dSdy)
        '''

        # nan/inf issues are with TS_grad
        dr_x = np.abs(self.TS_grad.alpha_dTdx / self.TS_grad.beta_dSdx)
        dr_y = np.abs(self.TS_grad.alpha_dTdy / self.TS_grad.beta_dSdy)

        # drop inf
        dr_x = xr.where(np.isinf(dr_x), np.nan, dr_x)
        dr_y = xr.where(np.isinf(dr_y), np.nan, dr_y)

        dr_x.name = 'density_ratio_x'
        dr_y.name = 'density_ratio_y'
        
        self.density_ratio = xr.merge([dr_x, dr_y])
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
        for key in ds.keys():
            ds_mean  = ds_mean.rename({key: key + '_ts_mean'})
            ds_std  = ds_std.rename({key: key + '_ts_std'})
        ds_stats = xr.merge([ds_mean, ds_std]).load()
        return ds_stats

    def get_T_S_bg_stats(self, save=False, load=False):
   
        if load:
            self.stats = xr.open_dataset(config.data_path() + self.case + 
                                      self.file_id + 'density_ratio_stats.nc')
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
                                       self.file_id + 'density_ratio_stats.nc')

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


m = plot_buoyancy_ratio('EXP10')
m.load_basics()
m.get_grad_T_and_S(save=True)
#m.get_density_ratio()
#m.get_T_S_bg_stats(save=True)

#m = plot_buoyancy_ratio('EXP10')
#m.get_T_S_bg_stats(load=True)
#m.plot_density_ratio()
