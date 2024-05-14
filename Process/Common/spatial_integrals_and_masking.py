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
        self.preamble = config.data_path() + case +  self.file_id
        self.path = config.data_path() + case + '/'

    def mask_by_ml(self, var):
        ''' mask variable by mixed layer depth mean '''

        # get time-mean grid_T
        kwargs = {'chunks': dict(time=1)} 
        ds = xr.open_dataset(self.preamble + 'grid_T.nc', **kwargs)

        # convert cell thickness to depths
        deps = ds_mean.e3t.cumsum('deptht')

        self.var_ml =  var.where(deps < ds.mldr10_3, drop=False)

    def cut_edges(self, rim=slice(10,-10), var):
        ''' cut forcing rim '''

        return var.isel(x=rim, y=rim)
        

    def domain_mean_ml_KE_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ area
        then integrate over domain and weight by volume
        '''

        # get data and domain_cfg
        self.mask_by_ml()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # load ice concentration
        icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                            chunks={'time_counter':10}).siconc

        # cut edges
        self.var_ml = self.cut_edges(self.var_ml)
        cfg = self.cut_edges(cfg)
        icemsk = self.cut_edges(icemsk)

        # get masks
        miz_msk = (icemsk_mean > threshold) & (icemsk_mean < (1 - threshold))
        ice_msk = icemsk_mean > (1 - threshold)
        oce_msk = icemsk_mean < threshold

        # mask by ice concentration
        var_ml_miz = self.var_ml.where(miz_msk)
        var_ml_ice = self.var_ml.where(ice_msk)
        var_ml_oce = self.var_ml.where(oce_msk)

        # define mean dims
        dims = ['x','y','deptht']

        # find volume of each partition
        area = cfg.e2t * cfg.e1t
        t_vol = area * e3t_mean_ml
        t_vol_miz = t_vol.where(miz_msk).sum(dim=dims)
        t_vol_ice = t_vol.where(ice_msk).sum(dim=dims)
        t_vol_oce = t_vol.where(oce_msk).sum(dim=dims)

        # calculate volume weighted mean
        t_volume = self.e3t_mean * area
        var_integ_miz = (var_ml_miz * t_volume).sum(dim=dims) / t_vol_miz
        var_integ_ice = (var_ml_ice * t_volume).sum(dim=dims) / t_vol_ice
        var_integ_oce = (var_ml_oce * t_volume).sum(dim=dims) / t_vol_oce

        # save
        with ProgressBar():
            fn = self.preamble + self.var_str + '_domain_integ_{}.nc'
            var_integ_miz.to_netcdf(fn.format('miz'))
            var_integ_ice.to_netcdf(fn.format('ice'))
            var_integ_oce.to_netcdf(fn.format('oce'))

    def horizontal_mean_ml_KE_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ
        then integrate in the horizontal and weight by area
        '''

        # get data and domain_cfg
        self.mask_by_ml()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # load ice concentration
        icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                            chunks={'time_counter':10}).siconc

        # cut edges
        self.var_ml = self.cut_edges(self.var_ml)
        cfg = self.cut_edges(cfg)
        icemsk = self.cut_edges(icemsk)

        # get masks
        miz_msk = (icemsk_mean > threshold) & (icemsk_mean < (1 - threshold))
        ice_msk = icemsk_mean > (1 - threshold)
        oce_msk = icemsk_mean < threshold

        # mask by ice concentration
        var_ml_miz = self.var_ml.where(miz_msk)
        var_ml_ice = self.var_ml.where(ice_msk)
        var_ml_oce = self.var_ml.where(oce_msk)

        # find area of each partition
        area = cfg.e2t * cfg.e1t
        area_miz = area.where(miz_msk).sum()
        area_ice = area.where(ice_msk).sum()
        area_oce = area.where(oce_msk).sum()

        # calculate lateral weighted mean
        var_integ_miz = (var_ml_miz * area).sum(dim=['x','y']) / area_miz
        var_integ_ice = (var_ml_ice * area).sum(dim=['x','y']) / area_ice
        var_integ_oce = (var_ml_oce * area).sum(dim=['x','y']) / area_oce

        # save
        with ProgressBar():
            fn = self.preamble + self.var_str + '_horizontal_integ_{}.nc'
            var_integ_miz.to_netcdf(fn.format('miz'))
            var_integ_ice.to_netcdf(fn.format('ice'))
            var_integ_oce.to_netcdf(fn.format('oce'))

