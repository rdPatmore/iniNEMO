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
        kwargs = {'chunks':{'deptht':-1, 'x':-1, 'y':-1}, 'decode_cf':False} 
        ke = xr.load_dataset(self.nc_preamble + 'KE.nc', **kwargs)
        mld = xr.load_dataset(self.nc_preamble + 'grid_T.nc', **kwargs
                             ).mldr10_3

        # conform time
        mld['time_counter'] = ke.time_counter
        #self.mld = self.mld.interp(time_counter=self.ke.time_counter)

        # reduce to mld
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

if __name__=='__main__':
    import time
    start = time.time()

    # cluster = LocalCluster(n_workers=5)
     #dask.config.set(scheduler='single-threaded')
     #client = Client(cluster)
     #client = Client(n_workers=5, memory_limit='64GB') # deals with memory issues
     
    nc_preamble = 'SOCHIC_PATCH_1h_20121209_20121211_'
    m = mld('TRD00', nc_preamble)
    print ('start')
    #m.find_KE_at_middepth()
    #print ('0')
    #m.find_KE_at_middepth()
    #print ('1')
    #m.find_uvel_at_middepth(depth=30)
    #print ('2')
    #m.find_vvel_at_middepth(depth=30)
    #print ('3')
    #m.find_momu_at_middepth(depth=30)
    #print ('4')
    #m.find_momv_at_middepth(depth=30)
    #print ('5')
    m.find_wvel_at_middepth(depth=30)
    print ('6')
    m.find_rho_at_middepth(depth=30)
    print ('7')

    end = time.time()
    print('time elapsed (minutes): ', (end - start)/60)
