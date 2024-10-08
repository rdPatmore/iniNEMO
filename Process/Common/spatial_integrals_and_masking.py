import xarray as xr
import config
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

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

    def mask_by_ml(self, save=False, cut=None, grid='grid_T'):

        ''' mask variable by mixed layer depth '''

        # get grid_X
        kwargs = {'chunks': dict(time_counter=100)} 
        ds = xr.open_dataset(self.raw_preamble + grid + '.nc', **kwargs)

        # get mld
        mld = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                             **kwargs).mldr10_3

        # cut edges of grid_T
        if cut:
            ds = ds.isel(x=cut[0], y=cut[1]) 
            mld = mld.isel(x=cut[0], y=cut[1]) 

        # convert cell thickness to depths
        if grid == 'grid_T':
            deps = ds.e3t.cumsum('deptht').load()
        elif grid == 'grid_W':
            deps = ds.e3w.cumsum('depthw').load()
            mld['time_counter'] = self.var.time_counter

        var_ml =  self.var.where(deps <= mld, drop=False)
        
        if save:
            with ProgressBar():
                fn = self.path + 'ProcessedVars/' + self.file_id + '{}_ml.nc'
                var_ml.to_netcdf(fn.format(self.var_str))

    def cut_edges(self, var, rim=slice(10,-10)):
        ''' cut forcing rim '''

        return var.isel(x=rim, y=rim)

    def save_mld_mid_pt(self, save=False):
        '''
        save mid point of mixed layer depth (chopped for bg norm)
        '''

        # get mld
        kwargs = {'chunks': dict(time_counter=1)} 
        mld = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                              **kwargs).mldr10_3

        # check index sizes match
        size_diff = self.check_index_size_diff(mld, self.var)
        if size_diff:
            mld = self.cut_edges(mld, rim=slice(size_diff, -size_diff))

        # get mid point
        mld_mid = mld / 2.0

        with ProgressBar():
            fn = self.path + 'ProcessedVars/' + self.file_id + 'ml_mid.nc'
            mld_mid.to_netcdf(fn)

    def extract_by_depth_at_mld_mid_pt(self, save=False):
        '''
        get data at mid point of mixed layer depth
        '''

        # get mld
        
        fn = self.path + 'ProcessedVars/' + self.file_id + 'ml_mid.nc'
        mld_mid = xr.open_dataarray(fn, chunks={'time_counter':100})

        # check index sizes match
        size_diff = self.check_index_size_diff(self.var, mld_mid)
        if size_diff:
            self.var = self.cut_edges(self.var, 
                                      rim=slice(size_diff, -size_diff))

        var_ml_mid = self.var.sel(deptht=mld_mid, method='nearest')

        if save:
            with ProgressBar():
                fn = self.path + 'ProcessedVars/' + self.file_id + \
                    '{}_ml_mid.nc'
                var_ml_mid.to_netcdf(fn.format(self.var_str))
        
    def check_index_size_diff(self, ds_a, ds_b):
        ''' check index sizes of two datasets '''

        x_diff = int((ds_a.x.size - ds_b.x.size) / 2)
        y_diff = int((ds_a.y.size - ds_b.y.size) / 2)

        # check rim differences in x and y match
        if x_diff == y_diff:
            size_diff = x_diff
        else:
            print ('WARNING: x and y are being cut incorrectly')

        return size_diff

    def get_domain_vars_and_cut_rims(self):
        '''
        get cfg and icemsk and cut rims along with with var
        '''

        # get domain_cfg and ice concentration
        cfg = xr.open_dataset(self.path + 'Grid/domain_cfg.nc',
                              chunks=-1).squeeze()
        icemsk = xr.open_dataset(
                         self.path + 'RawOutput/' + self.file_id + 'icemod.nc',
                            chunks={'time_counter':1}).siconc

        # check index sizes
        size_diff = self.check_index_size_diff(cfg, self.var)

        # trim edges accounting to size mismatch
        var_rim = 10 - size_diff

        # cut edges
        self.var = self.cut_edges(self.var, rim=slice(var_rim, -var_rim))
        self.cfg = self.cut_edges(cfg)
        self.icemsk = self.cut_edges(icemsk)

    def mask_by_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ area

        Parameters
        ----------

        threshold: sea ice concentration for partition between ice, oce and miz
        '''

        var = self.var

        # ensure time conformance
        if not self.icemsk.time_counter.identical(var.time_counter):
            var['time_counter'] = self.icemsk.time_counter

        # get masks
        miz_msk = ((self.icemsk > threshold) & (self.icemsk < (1 - threshold)))
        ice_msk = (self.icemsk > (1 - threshold))
        oce_msk = (self.icemsk < threshold)

        # mask by ice concentration
        var_ml_miz = var.where(miz_msk)
        var_ml_ice = var.where(ice_msk)
        var_ml_oce = var.where(oce_msk)

        return var_ml_miz, var_ml_ice, var_ml_oce

    def get_ice_oce_masked_var(self):
        '''
        mask by ice region and save
        '''

        # mask
        var_ml_miz, var_ml_ice, var_ml_oce = self.mask_by_ice_oce_zones()

        # set variable names
        var_ml_miz.name = self.var_str + '_miz'
        var_ml_ice.name = self.var_str + '_ice'
        var_ml_oce.name = self.var_str + '_oce'

        # merge variables
        var_masked = xr.merge([var_ml_miz, var_ml_ice, var_ml_oce])

        # save
        with ProgressBar():
           fn = self.path + 'ProcessedVars/' + self.file_id + '_'  \
              + self.var_str + '_ice_oce_miz.nc'
           var_masked.to_netcdf(fn)

    def domain_mean_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ area
        then integrate over domain and weight by volume

        Parameters
        ----------

        threshold: sea ice concentration for partition between ice, oce and miz
        '''

        # mask by ice concentration
        var_ml_miz, var_ml_ice, var_ml_oce = self.mask_by_ice_oce_zones()

        # get e3t
        kwargs = {'chunks': -1}
        e3t = xr.open_dataset(self.raw_preamble + 'grid_T.nc', **kwargs).e3t
        e3t = self.cut_edges(e3t)

        # define mean dims
        dims = ['x','y','deptht']

        # find volume of each partition
        t_vol = e3t * self.cfg.e2t * self.cfg.e1t

        # calculate volume weighted mean
        var_integ_miz = var_ml_miz.weighted(t_vol).mean(dim=dims)
        var_integ_ice = var_ml_ice.weighted(t_vol).mean(dim=dims)
        var_integ_oce = var_ml_oce.weighted(t_vol).mean(dim=dims)

        # set variable names
        var_integ_miz.name = self.var_str + '_miz_weighted_mean'
        var_integ_ice.name = self.var_str + '_ice_weighted_mean'
        var_integ_oce.name = self.var_str + '_oce_weighted_mean'

        # merge variables
        var_integ = xr.merge([var_integ_miz,
                              var_integ_ice,
                              var_integ_oce])

        # save
        with ProgressBar():
           fn = self.path + 'TimeSeries/' + self.var_str + '_domain_integ.nc'
           var_integ.to_netcdf(fn)

    def horizontal_mean_ice_oce_zones(self, threshold=0.2, save=False):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ
        then integrate in the horizontal and weight by area
        '''

        # mask by ice concentration
        var_ml_miz, var_ml_ice, var_ml_oce = self.mask_by_ice_oce_zones()

        # find area of each partition
        area = self.cfg.e2t * self.cfg.e1t

        # calculate lateral weighted mean
        var_integ_miz = var_miz.weighted(area).mean(dim=['x','y'])
        var_integ_ice = var_ice.weighted(area).mean(dim=['x','y'])
        var_integ_oce = var_oce.weighted(area).mean(dim=['x','y'])

        # set variable names
        var_integ_miz.name = self.var_str + '_miz_weighted_mean'
        var_integ_ice.name = self.var_str + '_ice_weighted_mean'
        var_integ_oce.name = self.var_str + '_oce_weighted_mean'

        # merge variables
        self.var_integ = xr.merge([var_integ_miz.load(),
                                   var_integ_ice.load(),
                                   var_integ_oce.load()])
 
        # save
        if save:
            fn = self.path + 'TimeSeries/' +\
                 self.var_str + '_horizontal_integ.nc'
            self.var_integ.to_netcdf(fn)
 
