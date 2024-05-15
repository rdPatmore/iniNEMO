import xarray as xr
import config
from dask.diagnostics import ProgressBar

class integrals_and_masks(object):
    '''
    Integrated variables in the dimensions of choice
    '''

    def __init__(self, case, file_id, var, var_str):
        self.var = var
        self.var_str = var_str
        self.file_id = file_id
        self.path = config.data_path() + case + '/'
        self.raw_preamble = self.path + 'RawOutput/' + self.file_id

    def mask_by_ml(self):
        ''' mask variable by mixed layer depth mean '''

# TOO computationally heavy - save first!
        # get time-mean grid_T
        kwargs = {'chunks': dict(time_counter=1)} 
        ds = xr.open_dataset(self.raw_preamble + 'grid_T.nc', **kwargs)

        # convert cell thickness to depths
        deps = ds.e3t.cumsum('deptht')

        self.var_ml =  self.var.where(deps < ds.mldr10_3, drop=False)

    def cut_edges(self, var, rim=slice(10,-10)):
        ''' cut forcing rim '''

        return var.isel(x=rim, y=rim)
        
    def domain_mean_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ area
        then integrate over domain and weight by volume
        '''

        # get data and domain_cfg
        self.mask_by_ml()
        cfg = xr.open_dataset(self.path + 'Grid/domain_cfg.nc',
                              chunks=-1).squeeze()

        # load ice concentration
        icemsk = xr.open_dataset(
                         self.path + 'RawOutput/' + self.file_id + 'icemod.nc',
                            chunks={'time_counter':1}).siconc

        # cut edges
        self.var_ml = self.cut_edges(self.var_ml)
        cfg = self.cut_edges(cfg)
        icemsk = self.cut_edges(icemsk)

        # get masks
        miz_msk = ((icemsk > threshold) & (icemsk < (1 - threshold))).load()
        ice_msk = (icemsk > (1 - threshold)).load()
        oce_msk = (icemsk < threshold).load()

        # mask by ice concentration
        var_ml_miz = self.var_ml.where(miz_msk)
        var_ml_ice = self.var_ml.where(ice_msk)
        var_ml_oce = self.var_ml.where(oce_msk)

        # define mean dims
        dims = ['x','y','deptht']

        # get e3t
        #kwargs = {'chunks': dict(deptht=1)}
        kwargs = {'chunks': -1}
        e3t = xr.open_dataset(self.raw_preamble + 'grid_T.nc', **kwargs).e3t
        e3t = self.cut_edges(e3t)

        # find volume of each partition
        t_vol = e3t * cfg.e2t * cfg.e1t
        t_vol_miz = t_vol.where(miz_msk).sum(dim=dims).load()
        t_vol_ice = t_vol.where(ice_msk).sum(dim=dims).load()
        t_vol_oce = t_vol.where(oce_msk).sum(dim=dims).load()

        # calculate volume weighted mean
        #var_integ_miz = (var_ml_miz * t_vol).sum(dim=dims).load() / t_vol_miz
        #var_integ_ice = (var_ml_ice * t_vol).sum(dim=dims).load() / t_vol_ice
        #var_integ_oce = (var_ml_oce * t_vol).sum(dim=dims).load() / t_vol_oce
        var_integ_oce = (var_ml_oce).sum(dim=dims) / t_vol_oce
        print ('done')

        # set variable names
        var_integ_miz.name = self.var_str + '_miz_weighted_mean'
        var_integ_ice.name = self.var_str + '_ice_weighted_mean'
        var_integ_oce.name = self.var_str + '_oce_weighted_mean'

        ## merge variables
        #var_integ = xr.merge([var_integ_miz.load(),
        #                      var_integ_ice.load(),
        #                      var_integ_oce.load()])

        ## save
        #fn = self.path + 'TimeSeries/' + self.var_str + '_domain_integ.nc'
        #var_integ.to_netcdf(fn)
        fn = self.path + 'TimeSeries/' + self.var_str + '_domain_integ_{}.nc'
        with ProgressBar():
            var_integ_oce.to_netcdf(fn.format('oce'))

    def horizontal_mean_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ
        then integrate in the horizontal and weight by area
        '''

        # get data and domain_cfg
        if 'deptht' in self.var.dims:
            var = self.mask_by_ml()
        else:
            var = self.var
        cfg = xr.open_dataset(self.path + 'Grid/domain_cfg.nc',
                              chunks=-1).squeeze()

        # load ice concentration
        icemsk = xr.open_dataset(
                         self.path + 'RawOutput/' + self.file_id + 'icemod.nc',
                            chunks={'time_counter':1}).siconc

        # cut edges
        var = self.cut_edges(var)
        cfg = self.cut_edges(cfg)
        icemsk = self.cut_edges(icemsk)

        # get masks
        miz_msk = ((icemsk > threshold) & (icemsk < (1 - threshold))).load()
        ice_msk = (icemsk > (1 - threshold)).load()
        oce_msk = (icemsk < threshold).load()

        # mask by ice concentration
        var_miz = var.where(miz_msk)
        var_ice = var.where(ice_msk)
        var_oce = var.where(oce_msk)

        # find area of each partition
        area = cfg.e2t * cfg.e1t
        dims= ['x','y']
        area_miz = area.where(miz_msk).sum(dim=dims).load()
        area_ice = area.where(ice_msk).sum(dim=dims).load()
        area_oce = area.where(oce_msk).sum(dim=dims).load()


        # calculate lateral weighted mean
        var_integ_miz = (var_miz * area).sum(dim=['x','y']) / area_miz
        var_integ_ice = (var_ice * area).sum(dim=['x','y']) / area_ice
        var_integ_oce = (var_oce * area).sum(dim=['x','y']) / area_oce

        # set variable names
        var_integ_miz.name = self.var_str + '_miz_weighted_mean'
        var_integ_ice.name = self.var_str + '_ice_weighted_mean'
        var_integ_oce.name = self.var_str + '_oce_weighted_mean'

        # merge variables
        var_integ = xr.merge([var_integ_miz.load(),
                              var_integ_ice.load(),
                              var_integ_oce.load()])
 
        # save
        fn = self.path + 'TimeSeries/' +\
             self.var_str + '_horizontal_integ.nc'
        var_integ.to_netcdf(fn)
 
