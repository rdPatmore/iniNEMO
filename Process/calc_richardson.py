import xarray as xr
import config
import numpy as np
import matplotlib.pyplot as plt
import iniNEMO.Process.calc_vorticity as vort
import dask
import gfuncs

# Let's start simple with a LocalCluster that makes use of all the cores and RAM we have on a single machine
from dask.distributed import Client, LocalCluster

class richardson(object):

    def __init__(self, model, nc_preamble):
        self.chunks = {'time_counter':1,'x':50,'y':50}
        self.chunks = {'time_counter':1}
        self.path = config.data_path() + model
        self.nc_preamble = self.path + '/' + nc_preamble
        #self.dsu = xr.open_dataset(self.path + 
        #               '/SOCHIC_PATCH_3h_20121209_20130331_grid_U.nc',
        #                chunks=self.chunks)
        #self.dsv = xr.open_dataset(self.path + 
        #               '/SOCHIC_PATCH_3h_20121209_20130331_grid_V.nc',
        #                chunks=self.chunks)
        #self.area = xr.open_dataset(self.path + 
        #                '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc',
        #                ).area
        self.cfg = xr.open_dataset(self.path + '/domain_cfg.nc',
                        )
                        #chunks={'x':50,'y':50} )

        # remove halo
        self.cfg  = self.cfg.isel(x=slice(1,-1), y=slice(1,-1)).squeeze()
        #self.area = self.area.isel(x=slice(1,-1), y=slice(1,-1))
        #self.dsu  = self.dsu.isel(x=slice(1,-1), y=slice(1,-1))
        #self.dsv  = self.dsv.isel(x=slice(1,-1), y=slice(1,-1))

    def buoyancy_gradients(self, save=False):
        ''' calculate the modulus square buoyancy gradient '''
         
        g = 9.81
        rho_0 = 1026
        rho = xr.open_dataset(self.nc_preamble + '_rho.nc', 
                              chunks=self.chunks).rho

        # calculate buoyancy gradients
        b = g*(1-rho/rho_0) 

        dx = self.cfg.e1u.isel(x=slice(None,-1), y=slice(None,-1))
        dy = self.cfg.e1v.isel(x=slice(None,-1), y=slice(None,-1))
        bx = b.diff('x', label='lower').isel(y=slice(None,-1)) / dx
        by = b.diff('y', label='lower').isel(x=slice(None,-1)) / dy
        bx.name = 'bx'
        by.name = 'by'
        bg = xr.merge([bx,by])

        if save:
            bg.to_netcdf(self.nc_preamble + '_bg.nc')

    def buoyancy_gradient_mod_squared(self, save=False):
        ''' calculate the modulus square buoyancy gradient '''
         
        bg = xr.open_dataset(self.nc_preamble + '_bg.nc', chunks=self.chunks)

        b_mod2 = bg.bx**2 + bg.by**2

        if save:
            bg_mod2.name = 'bg_mod2'
            bg_mod2.to_netcdf(self.nc_preamble + '_bg_mod2.nc')
        return b_mod2

    def format_N2(self, save=False):
        ''' format N2 to conform with rho '''

        rho = xr.open_dataset(self.nc_preamble + '_rho.nc',
                       chunks=self.chunks, decode_cf=False).rho
        N2 = xr.open_dataset(self.nc_preamble + '/_grid_W.nc',
                       chunks=self.chunks, decode_cf=False).bn2

        N2 = N2.interp(depthw=rho.deptht)#, time_counter=rho.time_counter)
        N2 = N2.drop('depthw')

        # convert time units for interpolation 
        #N2['time_counter'] = (N2.time_counter -
        #                      np.datetime64('1970-01-01 00:00:00')
        #                     ).astype(np.int64)
        #interp_time = (rho_time -
        #               np.datetime64('1970-01-01 00:00:00')
        #              ).astype(np.int64)
        # interpolate
        N2 = N2.interp(time_counter=rho.time_counter)

        # convert time units back to datetime64
        #N2['time_counter'] = N2.time_counter / 1e9 
        unit = "seconds since 1900-01-01 00:00:00"
        N2.time_counter.attrs['units'] = unit
        N2 = xr.decode_cf(N2.to_dataset()).bn2
        if save:
            N2.name = 'bn2'
            N2.to_netcdf(self.path + '/N2_conform.nc')
        return format_N2
    
    def merge_ri_components(self):
        f2 = self.cfg.ff_t**2
        N2 = self.format_N2()
        b_mod2 = self.buoyancy_gradient_mod_squared()
        #N2 = xr.open_dataarray(self.path + '/N2_conform.nc', chunks=self.chunks)
        #b_mod2 = xr.open_dataarray(self.path + '/b_grad_mod2.nc',
        #                          chunks=self.chunks)
        f2 = f2.isel(x=slice(None,-1), y=slice(None,-1))
        N2 = N2.isel(x=slice(1   ,-2), y=slice(1   ,-2))
        print (' ')
        print (f2)
        print (' ')
        print (N2)
        print (' ')
        print (b_mod2)
        print (' ')
        b_mod2 = b_mod2.reset_coords(['nav_lat','nav_lon'], drop=True)
        #f2, N2, b_mod2 = xr.align(f2,N2,b_mod2)
        merged_ri = xr.merge([f2,N2,b_mod2])
        print (merged_ri)

        merged_ri.to_netcdf(self.path + '/merged_ri.nc')


    def balanced_richardson_number(self, save=False):
        ''' calculate balanced richardson number '''
    
        #f2 = vort.vorticity('EXP08').planetary_vorticity(save=False)**2
        f2 = self.cfg.ff_t**2
        N2 = self.format_N2()
        b_mod2 = self.buoyancy_gradient_mod_squared()
        #f2 = xr.open_dataarray(self.path + '/cori.nc', chunks=self.chunks)**2
        #f2 = self.cfg.ff_t**2
        #N2 = xr.open_dataarray(self.path + '/N2_conform.nc', chunks=self.chunks)
        #b_mod2 = xr.open_dataarray(self.path + '/bg_mod2.nc',
                                   #chunks=self.chunks)
        f2 = f2.isel(x=slice(None,-1), y=slice(None,-1))
        N2 = N2.isel(x=slice(1   ,-2), y=slice(1   ,-2))
        #b_mod2 = b_mod2.reset_coords(['nav_lat','nav_lon'], drop=True)
        #f2, N2, b_mod2 = xr.align(f2,N2,b_mod2)
      
        #print (' ')
        #print (f2)
        #print (' ')
        #print (N2)
        #print (' ')
        m_ri = xr.merge([f2,N2,b_mod2])
        #print (b_mod2)
        #print (' ')
        #merged_ri = xr.merge([f2,N2,b_mod2])
        #print (merged_ri)
        #m_ri = xr.open_dataset(self.path + '/merged_ri.nc',
        #                       chunks=self.chunks)



        ## remove more edges to account for gradients below
        #f2 = f2.isel(x=slice(None,-1), y=slice(None,-1))
        #N2 = N2.isel(x=slice(1   ,-2), y=slice(1   ,-2))
        #print (f2)
        #print (N2.time_counter)
        #print (b_mod2.time_counter)
        
        # balanced richardson number
        Ri_b = m_ri.ff_t * m_ri.bn2 / m_ri.b_mod2
        Ri_b = Ri_b.where(np.abs(Ri_b) != np.inf)
        Ri_b = Ri_b.dropna(dim='time_counter', how='all')
        Ri_b = Ri_b.transpose('time_counter','deptht','y','x')

        print (Ri_b)
        if save:
            Ri_b.name = 'Ri_b'
            Ri_b.to_netcdf(self.nc_preamble + '_richardson_number.nc')

if __name__ == '__main__':
    import time
    #dask.config.set({'temporary_directory': 'Scratch'})
    dask.config.set(scheduler='single-threaded')
    #cluster = LocalCluster(n_workers=1)
    # explicitly connect to the cluster we just created
    #client = Client(cluster)

    nc_preamble = 'SOCHIC_PATCH_3h_20121209_20130331'
    m = richardson('EXP08', nc_preamble)
    start = time.time()
    m.buoyancy_gradients(save=True)
    #m.balanced_richardson_number(save=True)
    end = time.time()
    print(end - start)
