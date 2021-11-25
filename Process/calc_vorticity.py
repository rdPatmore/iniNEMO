import xarray as xr
import config
import numpy as np

class vorticity(object):

    def __init__(self, model):
        self.path = config.data_path() + model
        self.dsu = xr.open_dataset(self.path + 
                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_U.nc')
        self.dsv = xr.open_dataset(self.path + 
                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_V.nc')
        self.area = xr.open_dataset(self.path + 
                       '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').area
        self.cfg = xr.open_dataset(self.path + '/domain_cfg.nc')

        # remove halo
        self.cfg  = self.cfg.isel(x=slice(1,-1), y=slice(1,-1))
        self.area = self.area.isel(x=slice(1,-1), y=slice(1,-1))
        self.dsu  = self.dsu.isel(x=slice(1,-1), y=slice(1,-1))
        self.dsv  = self.dsv.isel(x=slice(1,-1), y=slice(1,-1))

        # rename depth
        self.dsu = self.dsu.rename({'depthu':'depth'})
        self.dsv = self.dsv.rename({'depthv':'depth'})

    def planetary_vorticity(self):
        ''' calculate f at vorticity points'''

        omega = 7.2921e-5 
        lat = self.cfg.gphif      # latitude on vorticity points
        f = 2 * omega * np.sin(lat) # coriolis
        return f

    def relative_vorticity(self):
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
        print (' ')
        print (' ')
        print (udx)
        print (' ')
        print (' ')
        print (vdy)
        print (' ')
        print (' ')
        print (area_vort)
        print (' ')
        print (' ')
        zeta = (-udx + vdy) / area_vort

        return zeta

    def rossby_number(self, save=False):
        ''' calculate vorticity form of Ro '''
    
        zeta = self.relative_vorticity()
        f = self.planetary_vorticity().isel(x=slice(None,-1), y=slice(None,-1))

        Ro = zeta / f
        
        if save:
            Ro.name = 'Ro'
            Ro.to_netcdf(self.path + '/rossby_number.nc')

        return Ro

m = vorticity('EXP13')
m.rossby_number(save=True)
