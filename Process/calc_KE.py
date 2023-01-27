import xarray as xr
import config
import matplotlib.pyplot as plt

class KE(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_1h_20121209_20121211_'
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
        
        ds = ds.isel(x=slice(45,-45), y=slice(45,-45))
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
        self.TKE_oce.name = 'TKE_oce'

        TKE = xr.merge([self.TKE_ice, self.TKE_oce])
    
        if save:
            TKE.to_netcdf(self.preamble + 'TKE_oce_ice.nc')

    def calc_TKE_budget(self):
        ''' calculate terms of the tubulent kinetic energy tendency '''

        # load primes
        kwargs = {'decode_cf':False} 
        uvel_prime = xr.open_dataarray(self.preamble + 'uvel_mld_rey.nc', **kwargs)
        umom_prime = xr.open_dataset(self.preamble + 'momu_mld_rey.nc', **kwargs)
        kwargs = {'decode_cf':False} 
        vvel_prime = xr.open_dataarray(self.preamble + 'vvel_mld_rey.nc', **kwargs)
        vmom_prime = xr.open_dataset(self.preamble + 'momv_mld_rey.nc', **kwargs)

        # get products
        u_tke = uvel_prime * umom_prime
        #u_tke.to_netcdf(self.preamble + 'u_tke.nc')
        v_tke = vvel_prime * vmom_prime

        # regrid to t-pts
        uT_tke = (u_tke + u_tke.roll(x=1, roll_coords=False)) / 2
        vT_tke = (v_tke + v_tke.roll(y=1, roll_coords=False)) / 2

        for var in uT_tke.data_vars:
            uT_tke = uT_tke.rename({var:var.lstrip('u')})
        for var in vT_tke.data_vars:
            vT_tke = vT_tke.rename({var:var.lstrip('v')})
        print (uT_tke)
        print (' ')
        print (vT_tke)

        uT_tke = uT_tke.mean('time_counter')
        vT_tke = vT_tke.mean('time_counter')
        #uT_tke.to_netcdf(self.preamble + 'uT_tke.nc')
        # tke budget
        tke_budg = 0.5 * ( uT_tke + vT_tke ).load()

        # save
        tke_budg.to_netcdf(self.preamble + 'tke_budget.nc')

    def calc_MKE_budget(self):
        # load and slice
        umom = xr.open_dataset(self.preamble + 'momu_mld.nc')
        vmom = xr.open_dataset(self.preamble + 'momv_mld.nc')
        uvel = xr.open_dataset(self.preamble + 'uvel_mld.nc').uo
        vvel = xr.open_dataset(self.preamble + 'vvel_mld.nc').vo

        # drop time
        umom = umom.drop_vars(['time_instant','time_instant_bounds',
                              'time_counter_bounds'])
        vmom = vmom.drop_vars(['time_instant','time_instant_bounds',
                              'time_counter_bounds'])

        # regrid to t-pts
        uT_mom = (umom + umom.roll(x=1, roll_coords=False)) / 2
        vT_mom = (vmom + vmom.roll(y=1, roll_coords=False)) / 2
        uT_vel = (uvel + uvel.roll(x=1, roll_coords=False)) / 2
        vT_vel = (vvel + vvel.roll(y=1, roll_coords=False)) / 2

        for var in uT_mom.data_vars:
            uT_mom = uT_mom.rename({var:var.lstrip('u')})
        for var in vT_mom.data_vars:
            vT_mom = vT_mom.rename({var:var.lstrip('v')})

        # means
        umom_mean = uT_mom.mean('time_counter')
        vmom_mean = vT_mom.mean('time_counter')
        uvel_mean = uvel.mean('time_counter')
        vvel_mean = vvel.mean('time_counter')

        # mean KE
        MKE = 0.5 * ( ( uvel_mean * umom_mean ) +
                      ( vvel_mean * vmom_mean ) )

        # save
        MKE.to_netcdf(self.preamble + 'MKE_mld_budget.nc')

    def calc_z_TKE_budget(self):

        # load
        kwargs = {'decode_cf':True} 
        wvel_prime = xr.open_dataarray(self.preamble + 'wvel_mld_rey.nc', **kwargs).load()
        rho_prime = xr.open_dataset(self.preamble + 'rho_mld_rey.nc', **kwargs).rhop.load()
        #rho = xr.open_dataset(self.preamble + 'grid_T.nc', **kwargs).rhop.load()
        rho_prime['time_counter'] = wvel_prime.time_counter

        #print (wvel_prime)
        print (rho_prime.min())
        print (rho_prime.max())
        # get products
        g = 9.81
        rho_0 = 1000 # rho prime is sigma0 (already anom to 1000)
        b_flux = g * wvel_prime * rho_prime / rho_0
        print (wvel_prime.min().values)
        print (wvel_prime.max().values)
        print (rho_prime.min().values)
        print (rho_prime.max().values)
        print (b_flux.min().values)
        print (b_flux.max().values)
        b_flux_mean = b_flux.mean('time_counter')

        # save
        b_flux_mean.to_netcdf(self.preamble + 'b_flux_mld.nc')
        

    def calc_TKE_steadiness(self):
        ''' plot time series of TKE and dTKE/dt at mixed later depth '''
       
        kwargs = {'decode_cf':False} 
        uvel_prime = xr.load_dataarray(self.preamble + 'uvel_mld_rey.nc', **kwargs)
        vvel_prime = xr.load_dataarray(self.preamble + 'vvel_mld_rey.nc', **kwargs)

        # regrid to t-pts
        uvelT_prime = (uvel_prime + uvel_prime.roll(x=1, roll_coords=False)) / 2
        vvelT_prime = (vvel_prime + vvel_prime.roll(y=1, roll_coords=False)) / 2

        # tke components
        uvelT_prime_sqmean = (uvelT_prime ** 2).mean(['time_counter'])
        vvelT_prime_sqmean = (vvelT_prime ** 2).mean(['time_counter'])

        # domain averaged TKE
        TKE_timeseries = 0.5 * (uvelT_prime_sqmean +
                                vvelT_prime_sqmean).mean(['x','y'])

        # save
        TKE_timeseries.to_netcdf(self.preamble + 'TKE_mld.nc')
       

m = KE('TRD00')
m.calc_MKE_budget()
#m.calc_TKE_steadiness()
#m.calc_z_TKE_budget()
#m.calc_reynolds_terms()
#m.calc_TKE(save=True)
