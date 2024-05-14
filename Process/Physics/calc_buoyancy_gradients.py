import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import iniNEMO.Process.Common.model_object as model_object
import dask

dask.config.set(scheduler='single-threaded')

class buoyancy_gradients(object):

    def __init__(self, case):
        self.case = case
        self.model = model_object.model(case)
        file_id = '/SOCHIC_PATCH_15mi_20121209_20121211_'
        #file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.preamble = config.data_path() + case + file_id
        #self.model.ds = self.model.ds['grid_T']
        #self.model.ds['rho'] = xr.open_dataarray(config.data_path() + case + 
        #                                       '/rho.nc')

    def open_temp(self):
        ''' open t grid file '''
        self.ds = xr.open_mfdataset(config.data_path() + self.case + 
                                '/SOCHIC_PATCH_*_grid_T.nc',
                                 chunks={'time_counter':1})

    def open_as_ct_p(self):
        ''' open conservative temperature and absolute salinity '''

        ct = xr.open_dataset(config.data_path() + self.case + 
                                 '/conservative_temperature.nc',
                                 chunks={'time_counter':1})
        abs_s = xr.open_dataset(config.data_path() + self.case + 
                                 '/absolute_salinity.nc',
                                 chunks={'time_counter':1})
        p = xr.open_dataset(config.data_path() + self.case + 
                                 '/p.nc',
                                 chunks={'time_counter':1})
        self.ds = xr.merge([ct, abs_s, p])

    def mixed_layer_buoyancy_gradients(self):
        '''
        calculate mixed layer buoyancy gradient 
            - Note superior calc in calc_richardson.py
        '''

        # restrict data to mixed layer
        self.model.ds = self.model.ds.where(
                       self.model.ds.deptht < self.model.ds.mldr10_3, drop=True)


        mesh_mask = xr.open_dataset(config.data_path() + self.case + 
                                 '/mesh_mask.nc').squeeze('time_counter')
 
        # remove halo
        mesh_mask = mesh_mask.isel(x=slice(1,-1), y=slice(1,-1))

        # constants
        g = 9.81
        rho_0 = 1027

        # mesh
        dx = mesh_mask.e1t.isel(x=slice(None,-1))
        dy = mesh_mask.e2t.isel(y=slice(None,-1))

        # buoyancy gradient
        buoyancy = g * (1 - self.model.ds.rho / rho_0)
        buoyancy_gradient_x = buoyancy.diff('x') / dx
        buoyancy_gradient_y = buoyancy.diff('y') / dy

        buoyancy_gradient_x.name = 'bgx'
        buoyancy_gradient_y.name = 'bgy'

        # hack to reassign coords ( unknown why these partially dissapear ) 
        #buoyancy_gradient_x = buoyancy_gradient_x.assign_coords({
        #                      'nav_lon':alphax.nav_lon,
        #                      'nav_lat':alphax.nav_lat})
        #buoyancy_gradient_y = buoyancy_gradient_y.assign_coords({
        #                      'nav_lon':alphay.nav_lon,
        #                      'nav_lat':alphay.nav_lat})

        bg = xr.merge([buoyancy_gradient_x.isel(y=slice(1,None)),
                       buoyancy_gradient_y.isel(x=slice(1,None))])
        bg.to_netcdf(config.data_path() + self.case + 
                    '/buoyancy_gradients.nc')

    def get_mixed_layer_buoyancy_gradient(self):
        '''
        load bg_norm2 created in calc_richardson.py and restrict to mixed layer
        '''

        # load bg
        kwargs = {'chunks':{'time_counter':10} ,'decode_cf':True} 
        bg_mod2 = xr.open_dataset(self.preamble + 'bg_mod2.nc', **kwargs)
        bg_mod2 = bg_mod2.set_coords(['nav_lon','nav_lat']).bg_mod2

        # load mld
        mld = xr.open_dataset(self.preamble + 'grid_T.nc', **kwargs
                                   ).mldr10_3
        #mld = mld.isel(x=slice(2,-2), y=slice(2,-2))
        mld = mld.isel(x=slice(1,-1), y=slice(1,-1))

        # mask below mixed layer
        bg_mod2 = bg_mod2.where(bg_mod2.deptht < mld)

        return bg_mod2 ** 0.5
        

    def split_bg_into_ice_oce_zones(self, save=False):
        '''
        split buoyancy gradients into two variables
            - ice area
            - ocean area
        '''

        # load buoyancy gradients 
        bg_norm = self.get_mixed_layer_buoyancy_gradient()

        # load ice presence
        icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                            chunks={'time_counter':10}, decode_cf=True).siconc
        #icemsk['time_counter'] = icemsk.time_instant
        icemsk = icemsk.isel(x=slice(1,-1), y=slice(1,-1))

        print (icemsk.time_counter)
        print (bg_norm.time_counter)

        # ice mask
        bg_norm_ice = bg_norm.where(icemsk > 0)
        bg_norm_oce = bg_norm.where(icemsk == 0)
        print (bg_norm_ice)
        bg_norm_ice.name = 'bg_norm_ice'
        bg_norm_oce.name = 'bg_norm_oce'

        # merge partitions
        self.bg_norm_partition = xr.merge([bg_norm_ice, bg_norm_oce])

        # save
        if save:
            self.bg_norm_partition.to_netcdf(
                                           self.preamble + 'bg_norm_ice_oce.nc')

    def get_zoned_quantiles(self):
        ''' get quantiles for spatially partitioned variables '''

        # restrict size
        self.bg_norm_partition = self.bg_norm_partition.isel(
                       x=slice(43,-43),y=slice(43,-43))

        quantiles = [0.02, 0.05, 0.2, 0.5, 0.8, 0.95, 0.98]
        dims = ['deptht','x','y']
        quant = self.bg_norm_partition.quantile(quantiles, dims).load()
        
        quant.to_netcdf(self.preamble + 'bg_norm_ice_oce_quantile.nc')
        

    def buoyancy_gradient_glider_path(self):
        #import dask
        #dask.config.set(**{'array.slicing.split_large_chunks': True})
        self.glider_nemo = xr.open_dataset(config.data_path() + self.case + 
                                '/glider_nemo.nc').load()
                                 #chunks={'distance':1})

        print (self.glider_nemo)

        dx = 1000
        dCT_dx = self.glider_nemo.cons_temp.diff('distance') / dx
        dAS_dx = self.glider_nemo.abs_sal.diff('distance') / dx
        dCT_dx = dCT_dx.pad(glider_path=(0,1))
        dAS_dx = dAS_dx.pad(glider_path=(0,1))
        alpha = self.glider_nemo.alpha
        beta = self.glider_nemo.beta
        g=9.81
        bg = g * ( alpha * dCT_dx - beta * dAS_dx ) 


        # restrict to mixed layer depth
        bg = bg.where(bg.deptht < self.glider_nemo.mldr10_3, drop=True)

        bg.name = 'bg'

        bg.time_counter.encoding['dtype'] = np.float64
        comp = dict(zlib=False, complevel=6)
        encoding = {var: comp for var in bg.data_vars}

        bg.to_netcdf(config.data_path() + self.case + '/glider_nemo_bg.nc',
                     unlimited_dims='time_counter')
        #bg.to_netcdf(config.data_path() + self.case + '/glider_nemo_bg.nc',
        #             encoding=encoding, unlimited_dims='time_counter')

    def buoyancy_gradient_stats(self):
        ''' calculate mean and standard deviations of buoyancy gradients '''
        bg = xr.open_dataset(config.data_path() + self.case + 
                                 '/buoyancy_gradients.nc',
                                  chunks={'time_counter':1})
        self.open_temp()
        self.ds = self.ds.isel(x=slice(1,-2), y=slice(None,-1))
        depth3d = self.ds.e3t.cumsum('deptht')
        bg = xr.where(depth3d < self.ds.mldr10_3, bg, np.nan) 

        bg['bgx'] = np.abs(bg.bgx)
        bg['bgy'] = np.abs(bg.bgy)

        dbdx_mean = bg.bgx.mean(['x', 'y','deptht']).load()
        dbdy_mean = bg.bgy.mean(['x', 'y','deptht']).load()
        dbdx_std = bg.bgx.std(['x', 'y','deptht']).load()
        dbdy_std = bg.bgy.std(['x', 'y','deptht']).load()
        dbdx_quant = bg.bgx.quantile([0.1,0.5,0.9], ['x', 'y','deptht']).load()
        dbdy_quant = bg.bgy.quantile([0.1,0.5,0.9], ['x', 'y','deptht']).load()

        dbdx_mean.name = 'dbdx_mean'
        dbdy_mean.name = 'dbdy_mean'
        dbdx_std.name = 'dbdx_std'
        dbdy_std.name = 'dbdy_std'
        dbdx_quant.name = 'dbdx_quant'
        dbdy_quant.name = 'dbdy_quant'
   
        bg_stats = xr.merge([dbdx_mean, dbdy_mean, dbdx_std, dbdy_std,
                             dbdx_quant, dbdy_quant])
        bg_stats.to_netcdf(config.data_path() + self.case + 
                           '/buoyancy_gradient_stats.nc')

bg = buoyancy_gradients('TRD00')
bg.split_bg_into_ice_oce_zones()
bg.get_zoned_quantiles()
#bg.mixed_layer_buoyancy_gradients()
#bg.buoyancy_gradient_stats()
#bg.open_temp()
#bg.save_pressure()
#bg.save_absolute_salinity()
#bg.save_alpha_beta()
#bg.buoyancy_gradient_glider_path()
