import xarray as xr
import config
import matplotlib.pyplot as plt

class KE(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.preamble = config.data_path() + case +  self.file_id

    def get_basic_vars(self):
        kwargs = {'chunks':{'time_counter':100} ,'decode_cf':False} 
        #kwargs = {'chunks':'auto' ,'decode_cf':False} 
        self.u = xr.open_dataset(self.preamble + 'grid_U.nc', **kwargs).uo
        self.v = xr.open_dataset(self.preamble + 'grid_V.nc', **kwargs).vo
        self.w = xr.open_dataset(self.preamble + 'grid_W.nc', **kwargs).wo
        self.e3w = xr.open_dataset(self.preamble + 'grid_W.nc', **kwargs
                                   ).e3w
        self.mld = xr.open_dataset(self.preamble + 'grid_T.nc', **kwargs
                                   ).mldr10_3
        #print (self.mld)
        # conform time
        self.mld['time_counter'] = self.u.time_counter

    def drop_time(self):
        ''' remove time dimension of basic variables '''

        self.u = self.u.isel(time_counter=100)
        self.v = self.v.isel(time_counter=100)
        self.w = self.w.isel(time_counter=100)
        self.mld = self.mld.isel(time_counter=100)

    def plot_slice(self, var):
        ''' plot horzontal slice of variable '''

        var = var.isel(z=0)
        plt.pcolor(var)
        plt.colorbar()
        plot.show()

    def mask_mld(self):
        ''' mask below mixed layer depth '''

        self.u = self.u.where(self.u.depthu < self.mld)
        self.v = self.v.where(self.v.depthv < self.mld)
        self.w = self.w.where(self.w.depthw < self.mld)

    def remove_edges(self, ds):
        ''' remove rim '''
        
        ds = ds.isel(x=slice(10,-10), y=slice(10,-10))
        return ds

    def grid_to_T_pts(self, save=False):
        ''' put find velocities on T-points '''
        
        #get vars
        self.get_basic_vars()

        # mask mld
        self.mask_mld()

        # interpolate u and v
        self.uT = (self.u + self.u.roll(x=1, roll_coords=False)) / 2
        self.vT = (self.v + self.v.roll(y=1, roll_coords=False)) / 2

        # interpolate w
        w_weighted = self.w * self.e3w
        w_scale = self.e3w + self.e3w.roll(depthw=-1, roll_coords=False)
        self.wT = (w_weighted + w_weighted.roll(depthw=-1, roll_coords=False)) \
                  / w_scale 

        #naming
        self.uT.name = 'uT'
        self.vT.name = 'vT'
        self.wT.name = 'wT'

        def conform_z_coords(da, z, dep_str):
            da = da.assign_coords({'z':xr.DataArray(z, dims=dep_str)})
            return da.swap_dims({dep_str:'z'}).drop(dep_str)

        # fix depth coords
        z = xr.open_dataset(self.preamble + 'grid_T.nc').deptht.values
        self.uT = conform_z_coords(self.uT, z, 'depthu')
        self.vT = conform_z_coords(self.vT, z, 'depthv')
        self.wT = conform_z_coords(self.wT, z, 'depthw')

        #self.uT.to_netcdf(self.preamble + 'uT.nc')
        # merge into dataset
        vels_T = xr.merge([self.uT, self.vT, self.wT])
       
        if save:
            vels_T.to_netcdf(self.preamble + 'vels_Tpt.nc')

    def calc_reynolds_terms(self):
        ''' calculate spatial-mean and spatial-deviations of velocities '''

        # open T-pt velocities
        vels = xr.open_dataset(self.preamble + 'vels_Tpt.nc',
                               chunks={'time_counter':10})

        # open ice mask
        #icemsk = xr.open_dataset(self.preamble + 'icemod.nc', decode_cf=False,
        icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                                 chunks={'time_counter':10} ).icepres
        icemsk['time_counter'] = icemsk.time_instant
        print (icemsk)
        print (vels)

        # ice mask
        vel_ice = vels.where(icemsk > 0)
        vel_oce = vels.where(icemsk == 0)

        # reynolds
        self.vel_bar_ice = vel_ice.mean(['x','y'])
        self.vel_bar_oce = vel_oce.mean(['x','y'])

        # get primes
        self.vel_prime_ice = self.remove_edges(self.vel_bar_ice - vel_ice)
        self.vel_prime_oce = self.remove_edges(self.vel_bar_oce - vel_oce)

        print (self.vel_prime_ice)
        # get prime mean squared
        self.vel_prime_ice_sqmean = (self.vel_prime_ice ** 2).mean(['x','y'])
        self.vel_prime_oce_sqmean = (self.vel_prime_oce ** 2).mean(['x','y'])

    def calc_TKE(self, save=False):
        ''' get turbulent kinetic energy '''
    
        # TKE in ice covered region
        self.TKE_ice = 0.5 * ( self.vel_prime_ice_sqmean.uT + 
                               self.vel_prime_ice_sqmean.vT +
                               self.vel_prime_ice_sqmean.wT ).load()
        self.TKE_ice.name = 'TKE_ice'

        # TKE over open ocean
        self.TKE_oce = 0.5 * ( self.vel_prime_oce_sqmean.uT + 
                               self.vel_prime_oce_sqmean.vT +
                               self.vel_prime_oce_sqmean.wT ).load()
        self.TKE_ice.name = 'TKE_oce'

        TKE = xr.merge([self.TKE_ice, self.TKE_oce])
    
        if save:
            self.TKE_ice.to_netcdf(self.preamble + 'TKE_oce_ice.nc')

m = KE('EXP10')
m.calc_reynolds_terms()
m.calc_TKE(save=True)
