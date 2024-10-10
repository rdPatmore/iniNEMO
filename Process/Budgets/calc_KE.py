import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.diagnostics import ProgressBar

class KE(object):

    def __init__(self, case, file_id):
        self.file_id = file_id
        self.proc_preamble = config.data_path() + case + '/ProcessedVars' \
                           +  self.file_id
        self.raw_preamble = config.data_path() + case + '/RawOutput' \
                           +  self.file_id
        self.path = config.data_path() + case + '/'

    def get_basic_vars(self):
        kwargs = {'chunks':{'time_counter':100} ,'decode_cf':False} 
        #kwargs = {'chunks':'auto' ,'decode_cf':False} 
        self.u = xr.open_dataset(self.raw_preamble + 'grid_U.nc', **kwargs).uo
        self.v = xr.open_dataset(self.raw_preamble + 'grid_V.nc', **kwargs).vo
        self.w = xr.open_dataset(self.raw_preamble + 'grid_W.nc', **kwargs).wo
        self.e3w = xr.open_dataset(self.raw_preamble + 'grid_W.nc', **kwargs
                                   ).e3w
        self.mld = xr.open_dataset(self.raw_preamble + 'grid_T.nc', **kwargs
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
        z = xr.open_dataset(self.raw_preamble + 'grid_T.nc').deptht.values
        self.uT = conform_z_coords(self.uT, z, 'depthu')
        self.vT = conform_z_coords(self.vT, z, 'depthv')
        self.wT = conform_z_coords(self.wT, z, 'depthw')

        #self.uT.to_netcdf(self.preamble + 'uT.nc')
        # merge into dataset
        vels_T = xr.merge([self.uT, self.vT, self.wT])
       
        if save:
            with ProgressBar():
                vels_T.to_netcdf(self.proc_preamble + 'vels_Tpt.nc')

    def calc_reynolds_terms(self, threshold=0.2):
        ''' calculate spatial-mean and spatial-deviations of velocities '''

        # open domain config
        cfg = xr.open_dataset(self.path + 'Grid/domain_cfg.nc',
                              chunks=-1).squeeze()

        # open T-pt velocities
        vels = xr.open_dataset(self.proc_preamble + 'vels_Tpt.nc',
                               chunks={'time_counter':10})

        # open ice mask
        #icemsk = xr.open_dataset(self.preamble + 'icemod.nc', decode_cf=False,
        icemsk = xr.open_dataset(self.raw_preamble + 'icemod.nc',
                                 chunks={'time_counter':10} ).siconc
        icemsk['time_counter'] = vels.time_counter

        # get masks - the next few lines are duplicated in /Common/spatial...
        miz_msk = ((icemsk > threshold) & (icemsk < (1 - threshold))).load()
        ice_msk = (icemsk > (1 - threshold)).load()
        oce_msk = (icemsk < threshold).load()

        # mask by ice concentration
        vel_miz = vels.where(miz_msk)
        vel_ice = vels.where(ice_msk)
        vel_oce = vels.where(oce_msk)

        # find area of each partition
        area = cfg.e2t * cfg.e1t

        # calculate lateral weighted mean
        self.vel_bar_miz = vel_miz.weighted(area).mean(dim=['x','y'])
        self.vel_bar_ice = vel_ice.weighted(area).mean(dim=['x','y'])
        self.vel_bar_oce = vel_oce.weighted(area).mean(dim=['x','y'])

        # get primes
        self.vel_prime_miz = self.remove_edges(self.vel_bar_miz - vel_miz)
        self.vel_prime_ice = self.remove_edges(self.vel_bar_ice - vel_ice)
        self.vel_prime_oce = self.remove_edges(self.vel_bar_oce - vel_oce)

        # trim area
        area = self.remove_edges(area)

        # get prime mean squared
        self.vel_prime_miz_sqmean = (self.vel_prime_miz ** 2
                                    ).weighted(area).mean(['x','y'])
        self.vel_prime_ice_sqmean = (self.vel_prime_ice ** 2
                                    ).weighted(area).mean(['x','y'])
        self.vel_prime_oce_sqmean = (self.vel_prime_oce ** 2
                                    ).weighted(area).mean(['x','y'])

    def calc_TKE(self, save=False):
        ''' get turbulent kinetic energy '''
    
        # TKE in ice covered region
        self.TKE_ice = 0.5 * ( self.vel_prime_ice_sqmean.uT + 
                               self.vel_prime_ice_sqmean.vT +
                               self.vel_prime_ice_sqmean.wT ).load()
        self.TKE_ice.name = 'TKE_ice'

        # TKE in marginal ice zone
        self.TKE_miz = 0.5 * ( self.vel_prime_miz_sqmean.uT + 
                               self.vel_prime_miz_sqmean.vT +
                               self.vel_prime_miz_sqmean.wT ).load()
        self.TKE_miz.name = 'TKE_miz'

        # TKE over open ocean
        self.TKE_oce = 0.5 * ( self.vel_prime_oce_sqmean.uT + 
                               self.vel_prime_oce_sqmean.vT +
                               self.vel_prime_oce_sqmean.wT ).load()
        self.TKE_oce.name = 'TKE_oce'

        TKE = xr.merge([self.TKE_ice, self.TKE_miz, self.TKE_oce])
    
        if save:
            with ProgressBar():
                TKE.to_netcdf(self.proc_preamble + 'TKE_oce_miz_ice.nc')

    def calc_TKE_budget(self, depth_str='mld'):
        ''' 
        Caclulate turbulent kinetic energy tendency excluding
        the vertical buoyancy flux term.
        '''
        # set chunking
        #chunksu = {'time_counter':10}
        #chunksv = {'time_counter':10}
        #chunkst = {'time_counter':10}
        chunksu = 'auto'
        chunksv = 'auto'
        chunkst = 'auto'

        # get momentum budgets
        append = '_rey.nc'
        umom = xr.open_dataset(self.proc_preamble + 'momu' + append,
                               chunks=chunksu)
        vmom = xr.open_dataset(self.proc_preamble + 'momv' + append,
                               chunks=chunksv)

        # remove u and v from variable names for combining
        for var in umom.data_vars:
            umom = umom.rename({var:var.lstrip('u')})
        for var in vmom.data_vars:
            vmom = vmom.rename({var:var.lstrip('v')})

        # get velocities
        uvel = xr.open_dataset(self.proc_preamble + 'uvel' + append,
                               chunks=chunksu).uo
        vvel = xr.open_dataset(self.proc_preamble + 'vvel' + append,
                               chunks=chunksv).vo

        # get scale factors
        e3u = xr.open_dataset(self.raw_preamble + 'uvel.nc',
                              chunks=chunksu).e3u
        e3v = xr.open_dataset(self.raw_preamble + 'vvel.nc',
                              chunks=chunksv).e3v
        e3t = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                              chunks=chunkst)

        # use time that is consistent with grid_W
        e3t['time_counter'] = e3t.time_instant
        e3t = e3t.e3t # get var

        # get TKE
        TKE = self.KE(umom, vmom, uvel, vvel, e3u, e3v, e3t)
        TKE = TKE.mean('time_counter')

        # save
        with ProgressBar():
            TKE.to_netcdf(self.proc_preamble + 'TKE_budget.nc')

    def merge_vertical_buoyancy_flux(self):
        ''' add vertical buoyancy flux to TKE dataset '''

        kwargs = {'chunks': {'time_counter': 100}}
        TKE = xr.open_dataset(self.proc_preamble + 'TKE_budget.nc', **kwargs)
        b_flux = xr.open_dataarray(self.proc_preamble + 'b_flux_rey_mean.nc',
                                    **kwargs)

        # merge in buoyancy flux
        TKE['trd_bfx'] = b_flux

        with ProgressBar():
            TKE.to_netcdf(self.proc_preamble + 'TKE_budget_full.nc')


    def calc_MKE_budget(self, depth_str='mld'):

        # load and slice
        umom = xr.open_dataset(self.proc_preamble + 'momu_' + depth_str + '.nc')
        vmom = xr.open_dataset(self.proc_preamble + 'momv_' + depth_str + '.nc')
        uvel = xr.open_dataset(self.proc_preamble + 'uvel_' + depth_str + '.nc')
        vvel = xr.open_dataset(self.proc_preamble + 'vvel_' + depth_str + '.nc')
        e3t  = xr.open_dataset(self.raw_preamble + 'grid_T_' + depth_str + '.nc').e3t

        for var in umom.data_vars:
            umom = umom.rename({var:var.lstrip('u')})
        for var in vmom.data_vars:
            vmom = vmom.rename({var:var.lstrip('v')})

        # load and slice
        umom = umom.mean('time_counter')
        vmom = vmom.mean('time_counter')
        uvel = uvel.mean('time_counter')
        vvel = vvel.mean('time_counter')
        e3t  = e3t.mean('time_counter')

        MKE = self.KE(umom_mean, vmom_mean,
                      uvel_mean, vvel_mean, e3t_mean)

        # save
        MKE.to_netcdf(self.proc_preamble + 'MKE_' + depth_str + '_budget.nc')

    def calc_KE_budget(self, depth_str='mld'):
        # load and slice
        umom = xr.open_dataset(self.proc_preamble + 'momu_' + depth_str + '.nc')
        vmom = xr.open_dataset(self.proc_preamble + 'momv_' + depth_str + '.nc')
        uvel = xr.open_dataset(self.proc_preamble + 'uvel_' + depth_str + '.nc')
        vvel = xr.open_dataset(self.proc_preamble + 'vvel_' + depth_str + '.nc')
        e3t  = xr.open_dataset(self.raw_preamble + 'grid_T_' + depth_str + '.nc').e3t

        # drop time
        umom = umom.drop_vars(['time_instant','time_instant_bounds',
                              'time_counter_bounds'])
        vmom = vmom.drop_vars(['time_instant','time_instant_bounds',
                              'time_counter_bounds'])

        for var in umom.data_vars:
            umom = umom.rename({var:var.lstrip('u')})
        for var in vmom.data_vars:
            vmom = vmom.rename({var:var.lstrip('v')})

        # interpolate time
        e3t = e3t.interp(time_counter=umom.time_counter)
        #e3t = e3t.interp(time_counter=umom.time_counter.astype('float64'))
        e3t['time_counter']  = umom.time_counter

        # means
        time = '2012-12-09 01:00:00'
        umom_snap = umom.sel(time_counter=time)
        vmom_snap = vmom.sel(time_counter=time)
        uvel_snap = uvel.sel(time_counter=time)
        vvel_snap = vvel.sel(time_counter=time)
        e3t_snap = e3t.sel(time_counter=time)

        KE = self.KE(umom_snap, vmom_snap,
                     uvel_snap, vvel_snap, e3t_snap)

        # save
        KE.to_netcdf(self.proc_preamble + 'KE_' + depth_str + '_budget.nc')

    def KE(self, umom, vmom, uvel, vvel, e3u, e3v, e3t, chunks=-1):
        ''' 
        calculate KE - shared function for KE, MKE and TKE
   
        Inputs
        ------
        umom (xr.Dataset): u-momentum terms
        vmom (xr.Dataset): v-momentum terms
        uvel (xr.DataArray): u-velocity
        vvel (xr.DataArray): v-velocity

        Returns
        -------
        KE (xr.Dataset): kinetic energy tendency terms
        '''

        cfg  = xr.open_dataset(self.path + 'domain_cfg.nc',
                               chunks=chunks).squeeze()

        bu = cfg.e1u * cfg.e2u * e3u
        bv = cfg.e1v * cfg.e2v * e3v
        bt = cfg.e1t * cfg.e2t * e3t

        # Note: 2d variables are broadcast to 3d
        uke = uvel * umom * bu
        vke = vvel * vmom * bv

        # coordinate hack
        uke = uke.rename({'depthu':'deptht'})
        vke = vke.rename({'depthv':'deptht'})

        KE = 0.5 * ( uke + self.ip1(uke) + vke + self.jp1(vke) ) / bt

        return KE


    def im1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(x=-1, roll_coords=False)

    def ip1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(x=1, roll_coords=False)

    def jm1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(y=-1, roll_coords=False)

    def jp1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(y=1, roll_coords=False)

    def km1(self, var, dvar='deptht'):
        ''' rolling opperations: roll down '''

        return var.roll({dvar:-1}, roll_coords=False)

    def kp1(self, var, dvar='deptht'):
        ''' rolling opperations: roll up '''

        return var.roll({dvar:1}, roll_coords=False)

    def calc_cori_err(self):
        ''' calculate the gridding error arrising due to cori gridding'''

        uvel = xr.open_dataset(self.raw_preamble + 'grid_U.nc').uo
        vvel = xr.open_dataset(self.raw_preamble + 'grid_V.nc').vo
        e3u = xr.open_dataset(self.raw_preamble + 'grid_U.nc').uo
        e3v = xr.open_dataset(self.raw_preamble + 'grid_V.nc').vo
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc')
        ff_f = xr.open_dataset(self.path + 'domain_cfg.nc').ff_f

        uflux = uvel * cfg.e2u * e3u
        vflux = vvel * cfg.e1v * e3v

        f3_ne = (         ff_f  + self.im1(ff_f) + self.jm1(ff_f))
        f3_nw = (         ff_f  + self.im1(ff_f) + self.im1(self.jm1(ff_f))) 
        f3_se = (         ff_f  + self.jm1(ff_f) + self.im1(self.jm1(ff_f))) 
        f3_sw = (self.im1(ff_f) + self.jm1(ff_f) + self.im1(self.jm1(ff_f))) 
        
        uPVO = (1/12.0)*(1/cfg.e1u)*(1/e3u)*( 
                                            f3_ne      * vflux 
                                 + self.ip1(f3_nw) * self.ip1(vflux)
                                 +          f3_se  * self.jm1(vflux)
                                 + self.ip1(f3_sw) * self.ip1(self.jm1(vflux)) )
        
        vPVO = -(1/12.0)*(1/cfg.e2v)*(1/e3v)*(
                                          self.jp1(f3_sw) * self.im1(jp1(uflux))
                                        + self.jp1(f3_se) * self.jp1(uflux)
                                        +          f3_nw  * self.im1(uflux)
                                        +          f3_ne  * uflux )

        uPVO.to_netcdf(self.raw_preamble + 'utrd_pvo_bta.nc')
        vPVO.to_netcdf(self.raw_preamble + 'vtrd_pvo_bta.nc')

    def calc_rhoW(self):
        ''' get rho on w-pts '''

        # get variables
        T_path = self.raw_preamble + 'grid_T.nc'
        W_path = self.raw_preamble + 'grid_W.nc'
        rho    = xr.open_dataset(T_path, decode_times=False).rhop
        depthw = xr.open_dataset(W_path, decode_times=False).depthw

        # shift to w-pts
        rhoW = 0.5 * (rho + self.kp1(rho)) 

        # mask bottom layer
        bot_T = rho.deptht.isel(deptht=-1)
        rhoW = rhoW.where(rhoW.deptht != bot_T)

        # switch to w coords
        rhoW = rhoW.swap_dims({'deptht':'depthw'})
        rhoW = rhoW.assign_coords({'depthw':depthw})

        # use time that is consistent with grid_W
        rhoW['time_counter'] = rhoW.time_instant

        # save
        rhoW.to_netcdf(self.proc_preamble + 'rhoW.nc')

    def calc_z_KE_budget(self):
        ''' calculate the vertical buoyancy flux '''
        
        # get variables
        chunk = {'time_counter':1}
        rhoW  = xr.open_dataset(self.proc_preamble + 'rhoW.nc', chunks=chunk).rhop
        e3t  = xr.open_dataset(self.raw_preamble + 'grid_T.nc', chunks=chunk).e3t
        wvel = xr.open_dataset(self.raw_preamble + 'wvel.nc', chunks=chunk).wo
        e3w = xr.open_dataset(self.raw_preamble + 'wvel.nc', chunks=chunk).e3w

        # calc buoyancy flux on w-pts
        rho0 = 1026
        g = 9.81
        z_conv = g * (1 - (rhoW / rho0)) * wvel * e3w 
     
        # shift to t-pts
        z_convT = 0.5 * ( z_conv + self.km1(z_conv, dvar='depthw') )

        # mask surface layer
        surf_W = z_convT.depthw.isel(depthw=0)
        z_convT = z_convT.where(z_conv.depthw != surf_W)

        # switch to t coords
        z_convT = z_convT.swap_dims({'depthw':'deptht'})
        z_convT = z_convT.assign_coords({'deptht':e3t.deptht}) 

        # use time that is consistent with grid_W
        e3t['time_counter'] = e3t.time_instant

        # buoyancy flux
        b_flux = z_convT / e3t

        # save
        b_flux.name = 'b_flux'
        b_flux.to_netcdf(self.proc_preamble + 'b_flux.nc')

    def calc_z_TKE_budget(self):
        ''' calculate the vertical buoyancy flux '''
        
        # get variables
        chunk = {'time_counter':1}
        rhoW  = xr.open_dataset(self.proc_preamble + 'rhoW_rey.nc', chunks=chunk).rhop
        e3t  = xr.open_dataset(self.raw_preamble + 'grid_T.nc', chunks=chunk).e3t
        wvel = xr.open_dataset(self.proc_preamble + 'wvel_rey.nc', chunks=chunk).wo
        e3w = xr.open_dataset(self.raw_preamble + 'grid_W.nc', chunks=chunk).e3w

        # calc buoyancy flux on w-pts
        rho0 = 1026
        g = 9.81
        # RDP not sure this should have been the density anomaly
        z_conv = g * (1 - (rhoW / rho0)) * wvel * e3w 
        #z_conv = g * rhoW * wvel * e3w 
     
        # shift to t-pts
        z_convT = 0.5 * ( z_conv + self.km1(z_conv, dvar='depthw') )

        # mask surface layer
        surf_W = z_convT.depthw.isel(depthw=0)
        z_convT = z_convT.where(z_conv.depthw != surf_W)

        # switch to t coords
        z_convT = z_convT.swap_dims({'depthw':'deptht'})
        z_convT = z_convT.assign_coords({'deptht':e3t.deptht}) 

        # use time that is consistent with grid_W
        e3t['time_counter'] = e3t.time_instant

        # buoyancy flux
        b_flux = z_convT / e3t

        # mean
        b_flux = b_flux.mean('time_counter') 

        # save
        b_flux.name = 'b_flux_rey_mean'
        with ProgressBar():
            b_flux.to_netcdf(self.proc_preamble + 'b_flux_rey_mean.nc')

    def calc_z_MKE_budget(self):
        ''' calculate the vertical buoyancy flux '''
        
        # get variables
        rhoW  = xr.open_dataset(self.proc_preamble + 'rhoW_mean.nc').rhop
        e3t  = xr.open_dataset(self.proc_preamble + 'grid_T_mean.nc').e3t
        wvel = xr.open_dataset(self.proc_preamble + 'grid_W_mean.nc').wo
        e3w = xr.open_dataset(self.proc_preamble + 'grid_W_mean.nc').e3w

        # calc buoyancy flux on w-pts
        rho0 = 1026
        g = 9.81
        z_conv = g * (1 - (rhoW / rho0)) * wvel * e3w 
     
        # shift to t-pts
        z_convT = 0.5 * ( z_conv + self.km1(z_conv, dvar='depthw') )

        # mask surface layer
        surf_W = z_convT.depthw.isel(depthw=0)
        z_convT = z_convT.where(z_conv.depthw != surf_W)

        # switch to t coords
        z_convT = z_convT.swap_dims({'depthw':'deptht'})
        z_convT = z_convT.assign_coords({'deptht':e3t.deptht}) 

        # buoyancy flux
        b_flux = z_convT / e3t

        # save
        b_flux.name = 'b_flux_mean'
        b_flux.to_netcdf(self.proc_preamble + 'b_flux_mean.nc')


    def calc_TKE_steadiness(self):
        ''' plot time series of TKE and dTKE/dt at mixed later depth '''
       
        kwargs = {'decode_cf':False} 
        uvel_prime = xr.load_dataarray(self.proc_preamble + 'uvel_mld_rey.nc',
                                       **kwargs)
        vvel_prime = xr.load_dataarray(self.proc_preamble + 'vvel_mld_rey.nc',
                                       **kwargs)

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
        TKE_timeseries.to_netcdf(self.proc_preamble + 'TKE_mld.nc')
       
if __name__ == '__main__':
     dask.config.set(scheduler='single-threaded')
     file_id = '/SOCHIC_PATCH_30mi_20121209_20121211_'
     m = KE('TRD00', file_id)
     #m.calc_rhoW()
     #m.calc_z_KE_budget()
     #m.calc_KE_budget(depth_str='30')

     #m.calc_TKE_budget()
     #m.calc_z_TKE_budget()
     m.merge_vertical_buoyancy_flux()

     # get TKE step 1
     #m.grid_to_T_pts(save=True)

     # get TKE step 2
     #m.calc_reynolds_terms()
     #m.calc_TKE(save=True)

     #m.calc_MKE_budget(depth_str='30')
     #m.calc_z_TKE_budget()
     #m.calc_z_MKE_budget()
     #m.calc_TKE_steadiness()
