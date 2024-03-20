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

matplotlib.rcParams.update({'font.size': 8})

class plot_buoyancy_ratio(object):

    def __init__(self, case, subset=None, giddy_method=False):

        self.case = case
        self.subset = subset
        self.giddy_method = giddy_method
        if subset:
            self.subset_var = '_' + subset
        else:
            self.subset_var = ''

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.f_path  = config.data_path() + case
        self.raw_f_path  = self.f_path + '/RawOutput' + self.file_id 
        self.prc_f_path  = self.f_path +  '/ProcessedVars' + self.file_id 

    def subset_n_s(self, arr, loc='north'):
        if loc == 'north':
        #    ind = int((arr.nav_lat + 59.9858035).idxmin('y', skipna=True)[0])
        #    arr = arr.isel(y=slice(ind,None))
            arr = arr.where(arr.nav_lat>-59.9858036, drop=True)
        if loc == 'south':
        #    ind = int((arr.nav_lat + 59.9858035).idxmin('y', skipna=True)[0])
        #    arr = arr.isel(y=slice(None,ind))
            arr = arr.where(arr.nav_lat<-59.9858036, drop=True)
        return arr

    def assign_x_y_index(self, arr, shift=0):
        arr = arr.assign_coords({'x':np.arange(shift,arr.sizes['x']+shift),
                                 'y':np.arange(shift,arr.sizes['y']+shift)})
        return arr

    def load_basics(self):

        chunks = {'time_counter':1,'deptht':1}
        self.bg = xr.open_dataset(self.prc_f_path + 'bg.nc', chunks=chunks)
        self.T = xr.open_dataset(self.raw_f_path + 'grid_T.nc', 
                                 chunks=chunks).votemper
        self.S = xr.open_dataset(self.raw_f_path + 'grid_T.nc',
                                 chunks=chunks).vosaline
        self.alpha = xr.open_dataset(self.prc_f_path + 'alpha.nc',
                                     chunks=chunks).to_array().squeeze()
        self.beta = xr.open_dataset(self.prc_f_path + 'beta.nc',
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
        self.alpha = self.assign_x_y_index(self.alpha, shift=1)
        self.beta  = self.assign_x_y_index(self.beta, shift=1)
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
            - wfo   : surface freshwater fluxes
            - qt_oce: surface heat fluxes
        '''

        # load
        chunks = {'time_counter':1,'deptht':1}
        self.wfo    = xr.open_dataset(self.raw_f_path + 'grid_T.nc',
                                   chunks=chunks).wfo
        self.qt_oce = xr.open_dataset(self.raw_f_path + 'grid_T.nc',
                                      chunks=chunks).qt_oce

        # assign index for x and y for merging
        self.wfo    = self.assign_x_y_index(self.wfo)
        self.qt_oce = self.assign_x_y_index(self.qt_oce)

        # subset model
        if self.subset:
            self.wfo    = self.subset_n_s(self.wfo,    loc=self.subset) 
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

    def shift_points(self, var, direc, drop=False):
        ''' between c, u and v points by averaging neighbours '''

        # mean neighbors
        shifted = var.rolling({direc:2}).mean() 

        # move blank to end of arr
        shifted = shifted.roll({direc:-1}, roll_coords=False)

        # drop nans created by rolling
        if drop:
            shifted = shifted.dropna(direc)

        return shifted

    def diff(self, arr, direc='x'):
        ''' diff, pad and reset x-coords '''

        # get end pad
        pad = xr.zeros_like(arr.isel({direc:-1}))

        # diff
        diffed = arr.diff(direc, label='lower')

        # pad
        diffed_padded = xr.concat([diffed,pad], dim=direc)

        return diffed_padded
        

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
                dTx = self.T.diff('x').pad(x=(0,1), constant_value=0) # u-pts
                dTy = self.T.diff('y').pad(y=(0,1), constant_value=0) # v-pts
                dSx = self.S.diff('x').pad(x=(0,1), constant_value=0) # u-pts
                dSy = self.S.diff('y').pad(y=(0,1), constant_value=0) # v-pts
                
                dTdx = self.alpha * dTx / dx
                dTdy = self.alpha * dTy / dy
                dSdx = self.beta  * dSx / dx
                dSdy = self.beta  * dSy / dy
            else:
                # gradient of alpha and beta included
                rhoT = self.alpha * self.T
                rhoS = self.beta  * self.S

                dTdx = self.diff(rhoT, direc='x') / dx # u-pts
                dTdy = self.diff(rhoT, direc='y') / dy # v-pts
                dSdx = self.diff(rhoS, direc='x') / dx # u-pts
                dSdy = self.diff(rhoS, direc='y') / dy # v-pts

            # move from u/v-pts to vorticity-points 
            dTdx_f = self.shift_points(dTdx, 'y')
            dTdy_f = self.shift_points(dTdy, 'x')
            dSdx_f = self.shift_points(dSdx, 'y')
            dSdy_f = self.shift_points(dSdy, 'x')

            ## restore indexes
            ###dTdx_f['x'] = rhoT.x.isel(x=slice(None,-1))
            #dTdy_f['y'] = rhoT.y.isel(y=slice(None,-1))
            #dSdx_f['x'] = rhoS.x.isel(x=slice(None,-1))
            #dSdy_f['y'] = rhoS.y.isel(y=slice(None,-1))

            # lat lon on f-points
            nav_lon_u = self.shift_points(self.cfg.nav_lon, 'x')
            nav_lon_f = self.shift_points(nav_lon_u, 'y')
            nav_lat_u = self.shift_points(self.cfg.nav_lat, 'x')
            nav_lat_f = self.shift_points(nav_lat_u, 'y')
            nav_lon_f = nav_lon_f.isel(x=slice(1,-1),y=slice(1,-1))
            nav_lat_f = nav_lat_f.isel(x=slice(1,-1),y=slice(1,-1))

            # use f-point lat lons
            dTdx_f['nav_lon'] = nav_lon_f
            dTdx_f['nav_lat'] = nav_lat_f
            dTdy_f['nav_lon'] = nav_lon_f
            dTdy_f['nav_lat'] = nav_lat_f
            dSdx_f['nav_lon'] = nav_lon_f
            dSdx_f['nav_lat'] = nav_lat_f
            dSdy_f['nav_lon'] = nav_lon_f
            dSdy_f['nav_lat'] = nav_lat_f
            
            # get norms
            gradT = (dTdx_f**2 + dTdy_f**2)**0.5
            gradS = (dSdx_f**2 + dSdy_f**2)**0.5
           
            # name
            dTdx_f.name = 'alpha_dTdx'
            dTdy_f.name = 'alpha_dTdy'
            dSdx_f.name = 'beta_dSdx'
            dSdy_f.name = 'beta_dSdy'
            gradT.name = 'gradTrho'
            gradS.name = 'gradSrho'
            
            self.TS_grad = xr.merge([dTdx_f.load(), dTdy_f.load(),
                                     dSdx_f.load(), dSdy_f.load(),
                                     gradT.load(),  gradS.load()])
            
            # save
            if save:
                self.TS_grad.to_netcdf(config.data_path() + self.case + 
                                       self.file_id + 'TS_grad_10m' +
                                       self.subset_var + '.nc')

    def get_density_ratio(self, load=False, save=False):
        ''' 
        get  (alpha * dTdx)/ (beta * dSdx)
             (alpha * dTdy)/ (beta * dSdy)
 
        requires get_grad_T_and_S()
        '''

        # define save str for giddy method   
        if self.giddy_method:
            giddy_str = '_giddy_method'
        else:
            giddy_str = ''

        f = self.prc_f_path + 'density_ratio' + giddy_str + self.subset_var \
            + '.nc'

        if load:
            self.density_ratio = xr.open_dataset(f)

        else:

            # nan/inf issues are with TS_grad
            # density ratio vectors
            dr_x = np.abs(self.TS_grad.alpha_dTdx / self.TS_grad.beta_dSdx)
            dr_y = np.abs(self.TS_grad.alpha_dTdy / self.TS_grad.beta_dSdy)

            # 2d denstiy ratio, where 1 is dividing line between
            # T and S dominance
            # see Ferrari and Paparella ((2004)
            dT = self.TS_grad.alpha_dTdx + 1j * self.TS_grad.alpha_dTdy 
            dS = self.TS_grad.beta_dSdx  + 1j * self.TS_grad.beta_dSdy 
            Tu_comp = dT / dS

            # get complex argument (angle)
            # workaround :: np.angle is yet to be dask enabled
            # returns angle between 0 and 180 in radians
            def get_complex_arg(arr):
                return np.angle(arr)
            Tu_phi = np.abs(xr.apply_ufunc(get_complex_arg, Tu_comp,
                                    dask='parallelized'))

            # get the complex modulus
            # returns positive angle in radians
            Tu_mod = np.arctan(np.abs(Tu_comp))

            # drop inf
            dr_x = xr.where(np.isinf(dr_x), np.nan, dr_x)
            dr_y = xr.where(np.isinf(dr_y), np.nan, dr_y)

            dr_x.name = 'density_ratio_x'
            dr_y.name = 'density_ratio_y'
            Tu_phi.name   = 'density_ratio_2d_phi'
            Tu_mod.name   = 'density_ratio_2d_mod'
            
            self.density_ratio = xr.merge([dr_x, dr_y, Tu_phi, Tu_mod])

            # save
            if save:
                self.density_ratio.to_netcdf(f)

    def get_stats(self, ds):
        ''' 
           get model mean, std and quantile time_series
        '''

        ds_mean  = np.abs(ds).mean(['x','y'], skipna=True)
        ds_std   = np.abs(ds).std(['x','y'], skipna=True)
        chunks = dict(x=-1, y=-1)
        ds_quant   = np.abs(ds.chunk(chunks)).quantile([0.05, 0.95],
                     ['x','y'], skipna=True)

 
        # test for DataArray/Dataset
        if type(ds).__name__ == 'DataArray':

            def rename_da(da, append):
                ds  = da.to_dataset()
                key = da.name
                return ds.rename({key: key + append + self.subset_var})

            ds_mean  = rename_da(ds_mean, '_ts_mean')
            ds_std   = rename_da(ds_std,  '_ts_std')
            ds_quant = rename_da(ds_quant,'_ts_quant')

        if type(ds).__name__ == 'Dataset':

            def rename_keys(var, key, append):
                return var.rename({key: key + append + self.subset_var})

            for key in ds.keys():
                ds_mean  = rename_keys(ds_mean,  key, '_ts_mean')
                ds_std   = rename_keys(ds_std,   key, '_ts_std')
                ds_quant = rename_keys(ds_quant, key, '_ts_quant')

        print (ds_mean)

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
        f = self.prc_f_path + 'density_ratio_stats' + giddy_str \
             + self.subset_var + '.nc'

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
        f = self.prc_f_path + 'bg_and_surface_flux_stats' + \
            self.subset_var + '.nc'

        if load:
            self.stats = xr.open_dataset(f) 

        else:
            # get stats (require load_basics and load_surface_fluxes)
            bg_stats = self.get_stats(self.bg)
            wfo_stats = self.get_stats(self.wfo)
            qt_oce_stats = self.get_stats(self.qt_oce)

            self.stats = xr.merge([bg_stats, wfo_stats, qt_oce_stats])

            # save
            if save:
                self.stats.to_netcdf(f)

    def get_Tu_frac(self):
        '''
        get fraction of domain containing
            - alpha versus beta ocean
            - compensating versus constructive horizontal buoyancy gradients

        why is the Tu data a strange shape? 1 removed from y and 2 from x
        rather than 1 and 1
        '''

        # get area and regrid to vorticity points
        area = xr.open_dataset(self.raw_f_path + 'grid_T.nc').area
        area = self.assign_x_y_index(area)
        if self.subset:
            area = self.subset_n_s(area, loc=self.subset)
        area_u = self.shift_points(area,   'x')
        area_f = self.shift_points(area_u, 'y')
        area_f = area_f.isel(x=slice(1,-1),y=slice(1,-1)) # match with alpha
        
        # lat lon on f-points
        nav_lon_u = self.shift_points(area.nav_lon, 'x')
        nav_lon_f = self.shift_points(nav_lon_u, 'y')
        nav_lat_u = self.shift_points(area.nav_lat, 'x')
        nav_lat_f = self.shift_points(nav_lat_u, 'y')

        nav_lon_f = nav_lon_f.isel(x=slice(1,-1),y=slice(1,-1))
        nav_lat_f = nav_lat_f.isel(x=slice(1,-1),y=slice(1,-1))

        # restore lat lon
        area_f['nav_lon'] = nav_lon_f
        area_f['nav_lat'] = nav_lat_f

        Tu_phi = self.density_ratio.density_ratio_2d_phi#.drop(['x','y'])
        Tu_mod = self.density_ratio.density_ratio_2d_mod#.drop(['x','y'])

        # subset north south
        if self.subset:
        #    print ('')
        #    print ('Tu_phi')
            Tu_phi = self.subset_n_s(Tu_phi, loc=self.subset)
        #    print ('')
        #    print ('Tu_mod')
            Tu_mod = self.subset_n_s(Tu_mod, loc=self.subset)
        #    print ('')
        #    print ('area_f')

        # total area
        area_sum = area_f.sum(['x','y'])

        # Tu_phi
        Tu_compen = xr.where(Tu_phi<np.pi/4, area_f,0).sum(['x','y']) / area_sum
        Tu_constr = xr.where(Tu_phi>np.pi/4, area_f,0).sum(['x','y']) / area_sum

        # Tu_mod
        Tu_sal = xr.where(Tu_mod<np.pi/4, area_f,0).sum(['x','y']) / area_sum
        Tu_tem = xr.where(Tu_mod>np.pi/4, area_f,0).sum(['x','y']) / area_sum

        return Tu_compen, Tu_constr, Tu_sal, Tu_tem

    def get_sea_ice_presence_stats(self, save=False):
        ''' load sea ice and subset '''

        # load ice
        self.si = xr.open_dataset(self.raw_f_path + 'icemod.nc').icepres

        # subset
        if self.subset:
            self.si = self.subset_n_s(self.si, loc=self.subset)

        self.si = self.get_stats(self.si)

        if save:
            f = self.prc_f_path + 'sea_ice_presence_stats' \
              + self.subset_var + '.nc'
            self.si.to_netcdf(f)


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


#    def test_proj(self):
#        '''
#        test for AlbersEqualArea projection
#        '''
#        fig = plt.figure(figsize=(5.5, 6.5), dpi=300)
#
#        lonmin, lonmax = -2,2
#        latmin, latmax = -62,-58
#
#        gs0 = gridspec.GridSpec(ncols=2, nrows=1)
#        gs0.update(top=0.99, bottom=0.15, left=0.15, right=0.98, wspace=0.05)
#
#        x = np.arange(-20,-10)
#        y = np.arange(-10,10)
#        data = np.sin(x+y)
#
#        axs0, axs1 = [], []
#        for i in range(2):
#            axs0.append(fig.add_subplot(gs0[i],
#                     projection=ccrs.AlbersEqualArea(central_latitude=-15,
#                      standard_parallels=(-10,-10))))
#            axs0[i].set_extent([lonmin, lonmax, latmin, latmax], ccrs.PlateCarree())
#        p0 = axs0[0].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
#                            cmap=cmocean.cm.ice, vmin=0, vmax=1,
#                              transform=ccrs.PlateCarree())

    def plot_density_ratio_with_SI_Ro_and_bg_time_series(self):
        '''
        plot over time - buoyancy gradient mean (n,s,all)
                       - buoyancy gradient std (n,s,all)
                       - fresh water fluxes and wfo (n,s,all)
                       - density ratio
        plot map - Sea ice concentration
                 - Ro
        GridSpec
         a) 1 x 2
         a) 4 x 1
        '''
        
        fig = plt.figure(figsize=(5.5, 6.5), dpi=300)

        lonmin, lonmax = -4,4
        latmin, latmax = -64,-56

        gs0 = gridspec.GridSpec(ncols=2, nrows=1)
        gs1 = gridspec.GridSpec(ncols=1, nrows=4)
        gs0.update(top=0.99, bottom=0.65, left=0.15, right=0.98, wspace=0.05)
        gs1.update(top=0.45, bottom=0.07,  left=0.15, right=0.98, hspace=0.17)

        axs0, axs1 = [], []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i],
                     projection=ccrs.AlbersEqualArea(central_latitude=-60,
                      standard_parallels=(latmin,latmax))))
            axs0[i].set_extent([lonmin, lonmax, latmin, latmax], ccrs.PlateCarree())
#            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(4):
            axs1.append(fig.add_subplot(gs1[i]))

        def render(ax, var, c='k', ls='-'):
            ax.plot(self.stats.time_counter, self.stats[var + self.subset_var],
                    c=c, ls=ls, lw=0.8)

        def render_density_ratio(ax, var):
            var_mean = var + '_ts_mean'
            lower = self.stats[var_mean].where(self.stats[var_mean] < 1)
            upper = self.stats[var_mean].where(self.stats[var_mean] > 1)
            ax.fill_between(lower.time_counter, lower, 1,
                            edgecolor=None, color='teal')
            ax.fill_between(upper.time_counter, 1, upper,
                            edgecolor=None, color='tab:red')


        # load sea ice and Ro
        si = xr.open_dataset(config.data_path() + 
                     'EXP10/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc').siconc
        Ro = xr.open_dataset(config.data_path_old() + 
                     'EXP10/rossby_number.nc').Ro
        # plot sea ice
        halo=2
        si = si.sel(time_counter='2012-12-30 00:00:00', method='nearest')
        si = si.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        
#        p0 = axs0[0].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
#                              cmap=cmocean.cm.ice, vmin=0, vmax=1,
#                              transform=ccrs.PlateCarree())
        p0 = axs0[0].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
                            cmap=cmocean.cm.ice, vmin=0, vmax=1,
                              transform=ccrs.PlateCarree())
        #axs0[0].set_aspect('equal')
    
        # plot Ro
        Ro = Ro.sel(time_counter='2012-12-30 00:00:00', method='nearest')
        Ro = Ro.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        Ro = Ro.isel(depth=10)
        
#        # render
#        p1 = axs0[1].pcolor(Ro.nav_lon, Ro.nav_lat, Ro, shading='nearest',
#                              cmap=plt.cm.RdBu, vmin=-0.45, vmax=0.45,
#                              transform=ccrs.PlateCarree())
        p1 = axs0[1].pcolor(Ro.nav_lon, Ro.nav_lat, Ro, shading='nearest',
                              cmap=plt.cm.RdBu, vmin=-0.45, vmax=0.45,
                              transform=ccrs.PlateCarree())
        #axs0[1].set_aspect('equal')

        l = []
        colours = ['orange', 'purple', 'green']
        ls = ['-', '-', '-']
        for i, subset in enumerate([None, 'north', 'south']):
            lab_str = subset
            if lab_str == None: lab_str = 'all'
            
            # update region
            self.subset = subset
            if subset:
                self.subset_var = '_' + subset
            else:
                self.subset_var = ''
            print ('subset', subset)

            # get stats
            self.get_bg_and_surface_flux_stats(load=True)
            self.get_density_ratio(load=True)
            Tu_compen, Tu_constr, Tu_sal, Tu_tem = self.get_Tu_frac()

            #self.get_grad_T_and_S(load=True)

            # render Temperature contirbution
            render(axs1[0], 'norm_grad_b_ts_mean', c=colours[i], ls=ls[i])
            render(axs1[1], 'wfo_ts_mean', c=colours[i], ls=ls[i])

            # plot Tu angle
            axs1[2].plot(Tu_constr.time_counter, Tu_constr*100, c=colours[i],
                         ls=ls[i], lw=0.8)
            p, = axs1[3].plot(Tu_sal.time_counter, Tu_sal*100,
                     label=lab_str, c=colours[i], ls=ls[i], lw=0.8)
            l.append(p)
          

        axs1[0].legend(l, ['all', 'north', 'south'], loc='upper center',
                       bbox_to_anchor=(0.25, 1.2, 0.5, 0.3), ncol=3)


        # axes formatting
        axs0[0].set_ylabel('latitude')
        axs0[1].yaxis.set_ticklabels([])
        #for ax in axs0:
        #    ax.set_xlabel('longitude')
        #    ax.set_xlim([-3.7,3.7])
        #    ax.set_ylim([-63.8,-56.2])
        
#        # colour bars
#        pos = axs0[0].get_position()
#        cbar_ax = fig.add_axes([pos.x0, 0.56, pos.x1 - pos.x0, 0.02])
#        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
#        cbar.ax.text(0.5, -2.0, r'Sea Ice Concentration [-]', fontsize=8,
#                rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')
#   
#        pos = axs0[1].get_position()
#        cbar_ax = fig.add_axes([pos.x0, 0.56, pos.x1 - pos.x0, 0.02])
#        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
#        cbar.ax.text(0.5, -2.0, r'$\zeta / f$ [-]', fontsize=8,
#                rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')
#
       #axs0[0].text(0.1, 0.9, '2012-12-30', transform=axs0[0].transAxes, c='w')

        date_lims = (self.stats.time_counter.min(), 
                     self.stats.time_counter.max())
        
        for ax in axs1:
            ax.set_xlim(date_lims)
        for ax in axs1[:-1]:
            ax.set_xticklabels([])


        axs1[0].set_ylabel(r'$|\nabla \mathbf{b}|$' + '\n' + r'[s$^{-2}]$')
        axs1[1].set_ylabel(r'$Q^{fw}$' + '\n' +r'[kg m$^{-2}$ s$^{-1}$]')
        axs1[2].set_ylabel(r'$\phi^{comp}$' + '\narea [%]')
        axs1[3].set_ylabel(r'$Tu^{beta}$' + '\narea [%]')

        # align labels
        xpos = -0.11  # axes coords
        for ax in axs1:
            ax.yaxis.set_label_coords(xpos, 0.5)

        # date labels
        for ax in axs1:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        axs1[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        axs1[-1].set_xlabel('date')

        plt.savefig('density_ratio_with_SI_Ro_and_bg_time_series_with_flxs.png',
                    dpi=600)

    def plot_density_ratio_slice(self):
        '''
        plot 2 x 1 panels of Tu phi and Tu mod
        '''
        
        fig, axs = plt.subplots(1,2, figsize=(5.5,3.8))
        plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.2,
                            wspace=0.05)
       
        m.get_density_ratio(load=True)

        # get grid - dubious method...
        lat = xr.open_dataset(self.raw_f_path + 'grid_T.nc').nav_lat
        lon = xr.open_dataset(self.raw_f_path + 'grid_T.nc').nav_lon
        lat = lat.isel(x=slice(1,None), y=slice(1,-1))
        lon = lon.isel(x=slice(1,None), y=slice(1,-1))
        
        # get Tu data
        Tu_phi = self.density_ratio.density_ratio_2d_phi.drop(['x','y'])
        Tu_mod = self.density_ratio.density_ratio_2d_mod.drop(['x','y'])

        Tu_phi_t0 = Tu_phi.sel(time_counter='2012-12-30 00:00:00',
                               method='nearest')
        Tu_mod_t0 = Tu_mod.sel(time_counter='2012-12-30 00:00:00',
                               method='nearest')
        
        # plot
        p0 = axs[0].pcolor(lon, lat, Tu_phi_t0,
                           cmap=plt.cm.RdBu_r, vmin=0, vmax=np.pi/2)
        p1 = axs[1].pcolor(lon, lat, Tu_mod_t0,
                            cmap=plt.cm.RdBu_r, vmin=0, vmax=np.pi/2)

        for ax in axs:
            ax.set_aspect('equal')

        # colour bars
        pos = axs[0].get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.12, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -2.3, r'$\phi$', fontsize=8,
                rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')
   
        pos = axs[1].get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.12, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -2.3, r'$|R|$', fontsize=8,
                rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')

        # axes formatting
        axs[0].set_ylabel('latitude')
        axs[1].yaxis.set_ticklabels([])
        for ax in axs:
            ax.set_xlabel('longitude')
            ax.set_xlim([-3.7,3.7])

        plt.savefig('density_ratio_snapshot_dec.png', dpi=600)


if __name__ == '__main__':
    # this needs to be run in two rounds, saving intermediate files:
    #                           - 1st/2nd round
    def prep_data():
        for subset in [None, 'north', 'south']:
            m = plot_buoyancy_ratio('EXP10', subset=subset)
            m.load_basics()
        #    m.get_grad_T_and_S(save=True)  # first round
        #    m.load_surface_fluxes()
        #    m.get_bg_and_surface_flux_stats(save=True)
            m.get_grad_T_and_S(load=True)
            m.get_density_ratio(save=True)
            #m.get_T_S_bg_stats(save=True) # second round
    
    def plot():
        m = plot_buoyancy_ratio('EXP10')
        #m.plot_density_ratio_slice()
        m.plot_density_ratio_with_SI_Ro_and_bg_time_series()
    
    def save_sea_ice_stats():
            m = plot_buoyancy_ratio('EXP10', subset='south')
            m.load_basics()
            m.get_sea_ice_presence_stats(save=True)
    save_sea_ice_stats()
    #m = plot_buoyancy_ratio('EXP10')
    #m.load_basics()
    #m.get_grad_T_and_S()
