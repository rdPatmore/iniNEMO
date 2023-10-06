import xarray as xr
import config
import numpy as np
import gsw
import dask

#dask.config.set(scheduler='single-threaded')

class orca(object):
    ''' get model object and process '''
 
    def __init__(self):
        self.data_path = config.data_path() + 'ORCA/'
        self.ds = {}
        #self.grid_keys = ['_T', '_U', '_V', '_I']
        self.grid_keys = ['_I']
        for pos in self.grid_keys:
            self.ds[pos] = xr.open_mfdataset(self.data_path +
                        '/ORCA_PATCH_*' + pos + '.nc',
                        compat='override',coords='minimal',
                        chunks={'time_counter':10}, decode_cf=True)

    #def calc_mld(self, ref_depth=10,threshold=0.03):
    #    """
    #    Calculated Mixed Layer Depth
    #    Default threshold is 0.03 referenced to 10m
    #    Data should be in format depth,distance
    #    """
    #    
    #    mld[i]=(depth[(np.abs((density[:,i]-density[ref_depth,i ]))>=
    #                   threshold)].min())

    #def save_mld(self):
    #    ''' calc mixed layer depth and save '''
    #    mld = xr.DataArray(self.calc_mld(), 
    #                       coords={'distance': self.giddy.distance},
    #                       dims='distance')
    #    mld_ds = self.giddy.assign(mld=mld).drop(['salt','temp','dens'])
    #    mld_ds.to_netcdf(self.root + 'Giddy_2020/giddy_mld.nc')

    def save_area_mean_all(self):
        ''' save lateral mean of all data '''

        for grid in self.grid_keys:
            print ('mean :', grid)
            ds = self.ds[grid].mean(['X','Y']).load()
            for key in ds.keys():
                ds = ds.rename({key: key + '_mean'})
            ds.to_netcdf(self.data_path + 'ORCA_PATCH_mean' 
                        + grid + '.nc')

    def save_area_std_all(self):
        ''' save lateral standard deviation of all data '''

        for grid in self.grid_keys:
            print ('std :', grid)
            ds = self.ds[grid].std(['X','Y']).load()
            for key in ds.keys():
                ds = ds.rename({key: key + '_std'})
            ds.to_netcdf(self.data_path + 'ORCA_PATCH_std' 
                         + grid + '.nc')


m = orca()
m.save_area_mean_all()
m.save_area_std_all()
