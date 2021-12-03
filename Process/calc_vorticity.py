import xarray as xr
import config
import numpy as np
import matplotlib.pyplot as plt

# Let's start simple with a LocalCluster that makes use of all the cores and RAM we have on a single machine
from dask.distributed import Client, LocalCluster

class vorticity(object):

    def __init__(self, model, big_data=False):
        self.chunks = {'time_counter':1}
        self.path = config.data_path() + model
        self.big_data = big_data
        if not big_data:
            self.dsu = xr.open_dataset(self.path + 
                           '/SOCHIC_PATCH_3h_20121209_20130331_grid_U.nc',
                            chunks=self.chunks)
            self.dsv = xr.open_dataset(self.path + 
                           '/SOCHIC_PATCH_3h_20121209_20130331_grid_V.nc',
                            chunks=self.chunks)
            self.area = xr.open_dataset(self.path + 
                            '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc',
                            ).area
            self.cfg = xr.open_dataset(self.path + '/domain_cfg.nc',
                            )

            # remove halo
            self.cfg  = self.cfg.isel(x=slice(1,-1), y=slice(1,-1)).squeeze()
            self.area = self.area.isel(x=slice(1,-1), y=slice(1,-1))
            self.dsu  = self.dsu.isel(x=slice(1,-1), y=slice(1,-1))
            self.dsv  = self.dsv.isel(x=slice(1,-1), y=slice(1,-1))

            # rename depth
            self.dsu = self.dsu.rename({'depthu':'depth'})
            self.dsv = self.dsv.rename({'depthv':'depth'})

            ## reduce time
            #self.dsu  = self.dsu.isel(time_counter=slice(0,100))
            #self.dsv  = self.dsv.isel(time_counter=slice(0,100))
         

    def planetary_vorticity(self, save=False):
        ''' calculate f at vorticity points'''

        omega = 7.2921e-5 
        lat = np.deg2rad(self.cfg.gphif) # latitude on vorticity points
        self.f = 2 * omega * np.sin(lat)      # coriolis

        if save:
            self.f = self.f.load()
            self.f.name = 'cori'
            self.f.to_netcdf(self.path + '/cori.nc')

    def relative_vorticity(self, save=False):
        ''' calculate relative vorticty '''

        # alias variables 
        dx = self.cfg.e1u
        dy = self.cfg.e1v
        u = self.dsu.uo
        v = self.dsv.vo

        # calculate area on vorticity points
        # NOTE: roll_coords will soon default to False
        area00 = self.area
        area10 = self.area.roll(x=-1, roll_coords=False)
        area01 = self.area.roll(y=-1, roll_coords=False)
        area11 = self.area.roll(x=-1, y=-1, roll_coords=False)
        area_vort = ( (area00 + area01 + area10 + area11) / 4 ).isel(
                                     x=slice(None,-1), y=slice(None,-1))

        # calculate vorticity 
        udx = (u*dx).diff('y', label='lower').isel(x=slice(None,-1))
        vdy = (v*dy).diff('x', label='lower').isel(y=slice(None,-1))
        self.zeta = (-udx + vdy) / area_vort

        if save:
            self.zeta.name = 'zeta'
            self.zeta.to_netcdf(self.path + '/zeta.nc')

    def rossby_number(self, save=False):
        ''' calculate vorticity form of Ro '''
    
        if self.big_data:
            self.zeta = xr.open_dataarray(self.path + '/zeta.nc',
                        chunks=self.chunks)
            self.f = xr.open_dataarray(self.path + '/cori.nc')
        else:
            self.relative_vorticity()
            self.planetary_vorticity()

        self.f = self.f.isel(x=slice(None,-1),y=slice(None,-1))

        self.Ro = self.zeta / self.f
        
        if save:
            self.Ro.name = 'Ro'
            #encoding = {'Ro':  dict(zlib=True, complevel=5)}
            self.Ro.to_netcdf(self.path + '/rossby_number.nc')
            #                  encoding=encoding)

if __name__ == '__main__':
    cluster = LocalCluster(n_workers=8)
    # explicitly connect to the cluster we just created
    client = Client(cluster)

    m = vorticity('EXP10', big_data=True)
    #print ('pre cori')
    #m.planetary_vorticity(save=True)
    #m.relative_vorticity(save=True)
    m.rossby_number(save=True)
