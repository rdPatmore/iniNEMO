import xarray as xr
import config
from dask.distributed import Client, LocalCluster
import dask
import matplotlib.pyplot as plt

class mld(object):
    ''' class for mixed layer depth operations'''

    def __init__(self, model, nc_preamble):
        self.path = config.data_path() + model
        self.nc_preamble = self.path + '/' + nc_preamble

    def season_mean(ds):
        # Number of days in each month
        month_length = ds.time.dt.days_in_month
    
        # Calculate the weights 
        weights = (
            month_length.groupby("time.season") /
            month_length.groupby("time.season").sum() )
    
        # Test that the sum of the weights for each season is 1.0
        np.testing.assert_allclose(
                       weights.groupby("time.season").sum().values, np.ones(4))
    
        # Calculate the weighted average
        weighted_mean =  (ds * weights).groupby("time.season").sum(dim="time")
      
        return weighted_mean

    def calculate_seasonal_and_spatial_mean_mld(self):
        ''' mean mld over seasons and space '''

        # load mld
        kwargs = {'chunks':{'deptht':-1, 'x':-1, 'y':-1}}
        mld = xr.open_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3

        # horizontal mean
        #mld_timeseries = mld.mean(['x','y'])
        mld_timeseries = mld.quantile([0.01]).squeeze().values

        print (mld_timeseries)

        #plt.plot(mld_timeseries)
        #plt.show()


        #print (mld_timeseries.min().values)
        #print (mld_timeseries.max().values)
        print (mld_timeseries[0].values)
        print (mld_timeseries[0].time_counter.values)
        print (mld_timeseries[-1].values)
        print (mld_timeseries[-1].time_counter.values)
        


    def find_KE_at_middepth(self, depth=None):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'deptht':-1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        ke = xr.load_dataset(self.nc_preamble + 'KE.nc', **kwargs)
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3

        # conform time
        mld['time_counter'] = ke.time_counter

        # reduce to mld
        if depth:
            ke = ke.sel(deptht=depth, method='nearest').load()
            ke.to_netcdf(self.nc_preamble + 'KE_' + str(depth) + '.nc')
        else:
            ke = ke.sel(deptht=mld/2, method='nearest').load()
            ke.to_netcdf(self.nc_preamble + 'KE_mld.nc')

    def find_uvel_at_middepth(self, depth=None):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthu':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        uvel = xr.load_dataset(self.nc_preamble + 'grid_U.nc', **kwargs)
        uvel = uvel[['e3u','uo']]
        kwargs = {'chunks':{'deptht':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3
        # conform time
        mld['time_counter'] = uvel.time_counter

        # reduce to mld
        if depth:
            uvel = uvel.sel(depthu=depth, method='nearest').load()
            uvel.to_netcdf(self.nc_preamble + 'uvel_' + str(depth) + '.nc')
        else:
            uvel = uvel.sel(depthu=mld/2, method='nearest').load()
            uvel.to_netcdf(self.nc_preamble + 'uvel_mld.nc')

    def find_vvel_at_middepth(self, depth=None):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthv':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        vvel = xr.load_dataset(self.nc_preamble + 'grid_V.nc', **kwargs)
        vvel = vvel[['e3v','vo']]
        kwargs = {'chunks':{'deptht':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3
        # conform time
        mld['time_counter'] = vvel.time_counter

        # reduce to mld
        if depth:
            vvel = vvel.sel(depthv=depth, method='nearest').load()
            vvel.to_netcdf(self.nc_preamble + 'vvel_' + str(depth) + '.nc')
        else:
            vvel = vvel.sel(depthv=mld/2, method='nearest').load()
            vvel.to_netcdf(self.nc_preamble + 'vvel_mld.nc')

    def find_wvel_at_middepth(self):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthw':1}, 'decode_cf':False} 
        wvel = xr.open_dataset(self.nc_preamble + 'grid_W.nc', **kwargs).wo
        kwargs = {'decode_cf':False} 
        deptht = xr.open_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).deptht.load()
        mld = xr.open_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3.load()

        # interp to t-pts
        wvel = wvel.interp(depthw=deptht)
        wvel = wvel.drop('depthw')

        # conform time
        mld['time_counter'] = wvel.time_counter

        # reduce to mld
        wvel = wvel.sel(deptht=mld/2, method='nearest').load()

        # save
        wvel.to_netcdf(self.nc_preamble + 'wvel_mld.nc')

    def find_rho_at_middepth(self, depth=30):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'decode_cf':False} 
        rho = xr.open_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).rhop.load()
        mld = xr.open_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3.load()

        # reduce to mld
        rho = rho.sel(deptht=mld/2, method='nearest').load()

        # save
        rho.to_netcdf(self.nc_preamble + 'rho_mld.nc')

    def find_momu_at_middepth(self, depth=None):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthu':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        momu = xr.load_dataset(self.nc_preamble + 'momu.nc', **kwargs)
        kwargs = {'chunks':{'deptht':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3
        # conform time
        mld['time_counter'] = momu.time_counter

        # reduce to mld
        if depth:
            momu = momu.sel(depthu=depth, method='nearest').load()
            momu.to_netcdf(self.nc_preamble + 'momu_' + str(depth) + '.nc')
        else:
            momu = momu.sel(depthu=mld/2, method='nearest').load()
            momu.to_netcdf(self.nc_preamble + 'momu_mld.nc')

    def find_momv_at_middepth(self, depth=30):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthv':-1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        momv =  xr.load_dataset(self.nc_preamble + 'momv.nc', **kwargs)
        kwargs = {'chunks':{'deptht':-1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3
        # conform time
        mld['time_counter'] = momv.time_counter

        print (momv)
        # reduce to mld
        if depth:
            momv = momv.sel(depthv=depth, method='nearest').load()
            momv.to_netcdf(self.nc_preamble + 'momv_' + str(depth) + '.nc')
        else:
            momv = momv.sel(depthv=mld/2, method='nearest').load()
            momv.to_netcdf(self.nc_preamble + 'momv_mld.nc')

    def reduce_temporal_resolution(self, interval):
        if interval == '1h':
            split = [0]
            f = f.isel(time_counter=ds.time_counter.dt.minute.isin(split))
        elif interval == '30mi':
            split = [0,30]
            f = f.isel(time_counter=ds.time_counter.dt.minute.isin(split))

    def find_var_at_middepth(self, var, depth=30):

        # find name of depth dimension
        dims = list(var.dims.keys())
        for dim in dims:
            if dim[:5] == 'depth':
                depth_var = dim

        # reduce to mld
        if depth:
            var = var.sel({depth_var:depth}, method='nearest').load()
        else:
            kwargs = {'chunks':{'deptht':-1, 'x':-1, 'y':-1}, 'decode_cf':False}
            mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                                 ).mldr10_3
            # conform time
            mld['time_counter'] = var.time_counter

            var = var.sel(depthv=mld/2, method='nearest').load()
        return var

    def save_var_at_middepth(self, f_name, depth=30,
                             vars=None):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'decode_cf':False} 
        ds =  xr.open_dataset(self.nc_preamble + f_name + '.nc', **kwargs)
        if vars:
            ds = ds[vars]

        # find name of depth dimension
        dims = list(var.dims.keys())
        for dim in dims:
            if dim[:5] == 'depth':
                depth_var = dim

        # reduce to mld
        ds = self.find_var_at_middepth(ds, depth_var, depth)
        ds.to_netcdf(self.nc_preamble + f_name + '_' + str(depth) + '.nc')
        #if depth:
        #    var = var.sel({depth_var:depth}, method='nearest').load()
        #else:
        #    kwargs = {'chunks':{'deptht':-1, 'x':-1, 'y':-1}, 'decode_cf':False}
        #    mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
        #                         ).mldr10_3
        #    # conform time
        #    mld['time_counter'] = var.time_counter

        #    var = var.sel(depthv=mld/2, method='nearest').load()

    def reduce_depth_and_time_dims(self, f_name, depth_var, depth=30, ts='1h',
                                   vars=None):
        '''
        reduce data 
          - get vertical slice
          - reduce temporal resolution
        '''

        # load data
        kwargs = {'decode_cf':False} 
        ds = xr.open_dataset(self.nc_preamble + f_name + '.nc', **kwargs)
        if vars:
            ds = ds[vars]

        # reduce
        ds = self.find_var_at_middepth(ds, depth_var, depth)
        ds = self.reduce_temporal_resolution(ts)

        # save
        nc_preamble = 'SOCHIC_PATCH_' + ts + '_20121209_20121211_'
        ds.to_netcdf(nc_preamble + f_name + 'dep_' + str(depth) + '.nc')

    def reduce_uvel_vars(self):
        ''' get vels and e3u from grid_U'''

        # load data
        kwargs = {'decode_cf':False} 
        ds = xr.open_dataset(self.nc_preamble + 'grid_U.nc', **kwargs)

        # reduce
        ds = ds[['uo','e3u']]

        # save
        ds.to_netcdf(self.nc_preamble + 'uvel.nc')

    def reduce_vvel_vars(self):
        ''' get vels and e3v from grid_V'''

        # load data
        kwargs = {'decode_cf':False} 
        ds = xr.open_dataset(self.nc_preamble + 'grid_V.nc', **kwargs)

        # reduce
        ds = ds[['vo','e3v']]

        # save
        ds.to_netcdf(self.nc_preamble + 'vvel.nc')

    def reduce_wvel_vars(self):
        ''' get vels and e3w from grid_W'''

        # load data
        kwargs = {'decode_cf':False} 
        ds = xr.open_dataset(self.nc_preamble + 'grid_W.nc', **kwargs)

        # reduce
        ds = ds[['wo','e3w']]

        # save
        ds.to_netcdf(self.nc_preamble + 'wvel.nc')

    def get_all(self):
        ''' get reductions in time and depth for all TKE variables '''

        files = ['grid_T','momu','momv']
        for f in files:
            self.save_var_at_middepth(f, depth=30)
            self.reduce_depth_and_time_dims(f, depth=30, ts='30mi')
            self.reduce_depth_and_time_dims(f, depth=30, ts='1h')

        v_list = ['uo','e3u']
        self.save_var_at_middepth('grid_U', depth=30, vars=vlist)
        self.reduce_depth_and_time_dims('grid_U', depth=30, ts='30mi',
                                        vars=vlist)
        self.reduce_depth_and_time_dims('grid_U', depth=30, ts='1h', 
                                        vars=vlist)

        v_list = ['vo','e3v']
        self.save_var_at_middepth('grid_V', depth=30, vars=v_list)
        self.reduce_depth_and_time_dims('grid_V', depth=30, ts='30mi',
                                        vars=v_list)
        self.reduce_depth_and_time_dims('grid_V', depth=30, ts='1h',
                                        vars=v_list)

if __name__=='__main__':
    import time
    start = time.time()

    # cluster = LocalCluster(n_workers=5)
     #dask.config.set(scheduler='single-threaded')
     #client = Client(cluster)
     #client = Client(n_workers=5, memory_limit='64GB') # deals with memory issues
     
    nc_preamble = 'SOCHIC_PATCH_3h_20121209_20130331_'

    m = mld('EXP10', nc_preamble)
    print ('start')
    m.calculate_seasonal_and_spatial_mean_mld()
#    m.find_var_at_middepth('b_flux', 'deptht', depth=30)
    #m.find_var_at_middepth('grid_T', 'deptht', depth=30)
    #m.reduce_uvel_vars()
    #m.reduce_vvel_vars()
    #m.reduce_wvel_vars()
#    m.find_var_at_middepth('momu', 'depthu', depth=30)
#    print ('0')
#    m.find_KE_at_middepth(depth=30)
#    print ('1')
#    m.find_uvel_at_middepth(depth=30)
#    print ('2')
#    m.find_vvel_at_middepth(depth=30)
#    print ('3')
#    m.find_momu_at_middepth(depth=30)
#    print ('4')
#    m.find_momv_at_middepth(depth=30)
#    print ('5')

    end = time.time()
    print('time elapsed (minutes): ', (end - start)/60)
