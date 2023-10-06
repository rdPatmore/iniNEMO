import xarray as xr
import config
import numpy as np
import gsw
import dask

#dask.config.set(scheduler='single-threaded')

class glider(object):
    ''' get model object and process '''
 
    def __init__(self):
        self.root = config.root()

        self.giddy = xr.open_dataset(self.root + 
                    'Giddy_2020/sg643_linterp.nc')
        print (self.giddy)

    def calc_mld(self, ref_depth=10,threshold=0.03):
        """
        Calculated Mixed Layer Depth
        Default threshold is 0.03 referenced to 10m
        Data should be in format depth,distance
        """
        density = self.giddy.dens
        depth = self.giddy.depth

        mld=np.ndarray(len(density[1,:]))
        for i in range(len(density[1,:])):
            try: 
                mld[i]=(depth[(np.abs((density[:,i]-density[ref_depth,i ]))>=
                       threshold)].min())
            except ValueError:  #raised if `y` is empty.
                mld[i]=(np.nan)
        return mld

    def save_mld(self):
        ''' calc mixed layer depth and save '''
        mld = xr.DataArray(self.calc_mld(), 
                           coords={'distance': self.giddy.distance},
                           dims='distance')
        mld_ds = self.giddy.assign(mld=mld).drop(['salt','temp','dens'])
        mld_ds.to_netcdf(self.root + 'Giddy_2020/giddy_mld.nc')



m = glider()
print (m.giddy)
m.save_mld()
