import xarray as xr
import config
from dask.distributed import Client, LocalCluster
import dask

class mld(object):
    ''' class for mixed layer depth operations'''

    def __init__(self, model, nc_preamble):
        self.path = config.data_path() + model
        self.nc_preamble = self.path + '/' + nc_preamble

    def find_KE_at_middepth(self):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'deptht':1, 'x':50, 'y':50},'decode_cf':True} 
        #self.mld = xr.open_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
        #                           ).mldr10_3
        self.ke = xr.open_dataset(self.nc_preamble + 'KE.nc', **kwargs)

        # conform time
        #self.mld = self.mld.interp(time_counter=self.ke.time_counter)

        #yypself.ke = self.ke.sel(deptht=self.ke.mldr10_3/2, method='nearest')
        self.ke = self.ke.isel(deptht=30)

        self.ke.to_netcdf(self.nc_preamble + 'KE_mld.nc')

    def find_uvel_at_middepth(self):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthu':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        uvel = xr.load_dataset(self.nc_preamble + 'grid_U.nc', **kwargs).uo
        kwargs = {'chunks':{'deptht':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3
        # conform time
        mld['time_counter'] = uvel.time_counter

        # reduce to mld
        #uvel = uvel.interp(depthu=mld_mid)#.load()
        uvel = uvel.sel(depthu=mld/2, method='nearest').load()

        # save
        uvel.to_netcdf(self.nc_preamble + 'uvel_mld.nc')

    def find_vvel_at_middepth(self):
        ''' reduce data to the middle of the mixed layer depth '''
         
        # load data
        kwargs = {'chunks':{'depthv':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        vvel = xr.load_dataset(self.nc_preamble + 'grid_V.nc', **kwargs).vo
        kwargs = {'chunks':{'deptht':1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3
        # conform time
        mld['time_counter'] = vvel.time_counter

        # reduce to mld
        vvel = vvel.sel(depthv=mld/2, method='nearest').load()

        # save
        vvel.to_netcdf(self.nc_preamble + 'vvel_mld.nc')

    def find_momu_at_middepth(self):
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
        momu = momu.sel(depthu=mld/2, method='nearest').load()

        # save
        momu.to_netcdf(self.nc_preamble + 'momu_mld.nc')

    def find_momv_at_middepth(self):
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
        momv = momv.sel(depthv=mld/2, method='nearest').load()

        # save
        momv.to_netcdf(self.nc_preamble + 'momv_mld.nc')

if __name__=='__main__':
    import time
    start = time.time()

    # cluster = LocalCluster(n_workers=5)
     #dask.config.set(scheduler='single-threaded')
     #client = Client(cluster)
     #client = Client(n_workers=5, memory_limit='64GB') # deals with memory issues
     
    nc_preamble = 'SOCHIC_PATCH_1h_20121209_20121211_'
    m = mld('EXP90', nc_preamble)
    m.find_momv_at_middepth()

    end = time.time()
    print('time elapsed (minutes): ', (end - start)/60)
