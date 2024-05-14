import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.diagnostics import ProgressBar

class integrals_and_masks(object):
    '''
    Integrated variables in the dimensions of choice
    '''

    def __init__(self, case, file_id, var):
        self.var = var
        self.file_id = file_id
        self.preamble = config.data_path() + case +  self.file_id
        self.path = config.data_path() + case + '/'

    def mask_by_ml(self, var):
        ''' mask variable by mixed layer depth mean '''

        # get time-mean grid_T
        kwargs = {'chunks': dict(time=1)} 
        ds = xr.open_dataset(self.preamble + 'grid_T.nc', **kwargs)

        self.var_ml =  var.where(ds.e3t < ds.mldr10_3, drop=False)

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
                            chunks={'time_counter':10}, decode_cf=True).siconc

        # mask e3t by mean mixed layer depth
        e3t_mean_ml = self.mask_by_ml(self.e3t_mean)

        # cut edges
        cut=slice(10,-10)
        self.tke_mld = self.tke_mld.isel(x=cut,y=cut)
        cfg = cfg.isel(x=cut,y=cut)
        icemsk = icemsk.isel(x=cut,y=cut)
        e3t_mean_ml = e3t_mean_ml.isel(x=cut,y=cut)
        self.e3t_mean = self.e3t_mean.isel(x=cut,y=cut)

        # mean ice conc over time series
        icemsk_mean = icemsk.mean('time_counter')

        # get masks
        miz_msk = (icemsk_mean > threshold) & (icemsk_mean < (1 - threshold))
        ice_msk = icemsk_mean > (1 - threshold)
        oce_msk = icemsk_mean < threshold

        # mask by ice concentration
        tke_mld_miz = self.tke_mld.where(miz_msk)
        tke_mld_ice = self.tke_mld.where(ice_msk)
        tke_mld_oce = self.tke_mld.where(oce_msk)

        # find volume of each partition
        area = cfg.e2t * cfg.e1t
        t_vol = area * e3t_mean_ml
        t_vol_miz = t_vol.where(miz_msk).sum()
        t_vol_ice = t_vol.where(ice_msk).sum()
        t_vol_oce = t_vol.where(oce_msk).sum()

        # calculate volume weighted mean
        t_volume = self.e3t_mean * area
        tke_integ_miz = (tke_mld_miz * t_volume).sum() / t_vol_miz
        tke_integ_ice = (tke_mld_ice * t_volume).sum() / t_vol_ice
        tke_integ_oce = (tke_mld_oce * t_volume).sum() / t_vol_oce

        # save
        with ProgressBar():
            fn = self.preamble + 'TKE_budget_domain_integ_{}.nc'
            tke_integ_miz.to_netcdf(fn.format('miz'))
            tke_integ_ice.to_netcdf(fn.format('ice'))
            tke_integ_oce.to_netcdf(fn.format('oce'))

    def horizontal_mean_ml_KE_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ
        then integrate in the horizontal and weight by area
        '''

        # get data and domain_cfg
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # load ice concentration
        icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                            chunks={'time_counter':10}, decode_cf=True).siconc

        # cut edges
        cut=slice(10,-10)
        self.tke_mld = self.tke_mld.isel(x=cut,y=cut)
        cfg = cfg.isel(x=cut,y=cut)
        icemsk = icemsk.isel(x=cut,y=cut)

        # mean ice conc over time series
        icemsk_mean = icemsk.mean('time_counter')

        # get masks
        miz_msk = (icemsk_mean > threshold) & (icemsk_mean < (1 - threshold))
        ice_msk = icemsk_mean > (1 - threshold)
        oce_msk = icemsk_mean < threshold

        # mask by ice concentration
        tke_mld_miz = self.tke_mld.where(miz_msk)
        tke_mld_ice = self.tke_mld.where(ice_msk)
        tke_mld_oce = self.tke_mld.where(oce_msk)

        # find area of each partition
        area = cfg.e2t * cfg.e1t
        area_miz = area.where(miz_msk).sum()
        area_ice = area.where(ice_msk).sum()
        area_oce = area.where(oce_msk).sum()

        # calculate lateral weighted mean
        tke_integ_miz = (tke_mld_miz * area).sum(dim=['x','y']) / area_miz
        tke_integ_ice = (tke_mld_ice * area).sum(dim=['x','y']) / area_ice
        tke_integ_oce = (tke_mld_oce * area).sum(dim=['x','y']) / area_oce

        # save
        with ProgressBar():
            fn = self.preamble + 'TKE_budget_horizontal_integ_{}.nc'
            tke_integ_miz.to_netcdf(fn.format('miz'))
            tke_integ_ice.to_netcdf(fn.format('ice'))
            tke_integ_oce.to_netcdf(fn.format('oce'))

