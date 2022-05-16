import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np

class plot_buoyancy_ratio(object):

    def __init__(self, case, subset=''):

        self.case = case
        self.subset = subset

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.bg = xr.open_dataset(config.data_path() + case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc',
                             chunks='auto')
        self.gridT = xr.open_dataset(config.data_path() + case +
                              '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc',
                              chunks='auto')
        self.alpha = xr.open_dataset(config.data_path() + case +
                              '/SOCHIC_PATCH_3h_20121209_20130331_alpha.nc',
                              chunks='auto').to_array().squeeze()
        self.beta = xr.open_dataset(config.data_path() + case +
                              '/SOCHIC_PATCH_3h_20121209_20130331_beta.nc',
                              chunks='auto').to_array().squeeze()
        # name the arrays (this should really be done in model_object
        #                  where the data is made)
        self.alpha.name = 'alpha'
        self.beta.name = 'beta'

        self.cfg = xr.open_dataset(config.data_path() + case +
                                   '/domain_cfg.nc').squeeze()

        print ('a')

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
        print ('c')

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
        self.bg = self.bg.sel(deptht=10, time_counter='2013-01-01 00:00:00', method='nearest')
        self.gridT = self.gridT.sel(deptht=10, time_counter='2013-01-01 00:00:00', method='nearest')
        self.alpha = self.alpha.sel(deptht=10, time_counter='2013-01-01 00:00:00', method='nearest')
        self.beta = self.beta.sel(deptht=10, time_counter='2013-01-01 00:00:00', method='nearest')

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

    def get_grad_T_and_S(self, load=False, save=False):
        ''' get dCdx and dCdy where C=[T,S] '''

        if load:
            self.TS_grad = xr.open_dataset(config.data_path() + self.case +
                                    self.file_id +'TS_grad_10m.nc')
        else:
            dx = self.cfg.e1t
            dy = self.cfg.e2t
            dgridTx = self.gridT.diff('x').pad(x=(1,0),
                                      constant_value=1) # u-pts
            dgridTy = self.gridT.diff('y').pad(y=(1,0),
                                      constant_value=1) # v-pts
            
            dTdx = dgridTx.votemper / dx
            dTdy = dgridTy.votemper / dy
            dSdx = dgridTx.vosaline / dx
            dSdy = dgridTy.vosaline / dy

            # name
            dTdx.name = 'dTdx'
            dTdy.name = 'dTdy'
            dSdx.name = 'dSdx'
            dSdy.name = 'dSdy'
            
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

        dr_x = np.abs(self.alpha * self.TS_grad.dTdx) /      \
               np.abs(self.beta * self.TS_grad.dSdx)
        dr_y = np.abs(self.alpha * self.TS_grad.dTdy) /      \
               np.abs(self.beta * self.TS_grad.dSdy)
        dr_y = dr_y.drop(['time_counter','time_centered','time_instant'])

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

    def get_T_S_bg_stats(self):
        print ('0')
        self.get_grad_T_and_S(load=True)
        print ('1')
        self.get_density_ratio()
        print ('2')

        self.TS_grad = self.get_stats(self.TS_grad)
        print ('3')
        self.bg = self.get_stats(self.bg)
        print ('4')
        self.density_ratio = self.get_stats(self.density_ratio)
        print ('5')

        self.stats = xr.merge([self.TS_grad, self.bg, self.density_ratio])
        print ('6')
        print (self.stats)

    def plot_density_ratio(self):
        '''
        plot - buoyancy gradient
             - temperature gradient
             - salinity gradient
             - density ratio
        over time

        4 x 2 plot with columns of x and y components
        '''

        fig, axs = plt.subplots(2,4, figsize=(5.5,5.5))

        def render(ax, var):
            var_mean = var + '_ts_mean'
            var_std = var + '_ts_std'
            ax.fill_between(self.stats.time_counter,
                                self.stats[var_mean] - self.stats[var_std],
                                self.stats[var_mean] + self.stats[var_std],
                                edgecolor=None)
            ax.plot(self.stats.time_counter, self.stats[var_mean])

        var_list = ['dbdx','dbdy','dTdx','dTdy','dSdx','dSdy']
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                print (var_list[i,j], i, j)
                render(ax,var_list[i+j])


m = plot_buoyancy_ratio('EXP10')
m.get_grad_T_and_S(save=True)
#m.get_T_S_bg_stats()
