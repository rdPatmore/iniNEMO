import xarray as xr
import config
from dask.diagnostics import ProgressBar

class downsample(object):

    def __init__(self, case, file_id):
        self.file_id = file_id
        self.proc_preamble = config.data_path() + case + '/ProcessedVars/'
        self.raw_preamble = config.data_path() + case + '/RawOutput/'

    def downsample_time(self, grid_str, file_id_slice, slice_dist):
        ''' reduce time resolution and save '''

        # get data
        kwargs = {'chunks':'auto' ,'decode_cf':False} 
        fn = self.raw_preamble + self.file_id + grid_str + '.nc'
        ds = xr.open_dataset(fn, **kwargs)

        # choose offset - TODO: it should be possible to use a functional form
        if slice_dist == 2:
            offset = 1
        elif slice_dist == 4:
            offset = 3
        else:
            offset = None
            print ('WARNING: no offset defined')

        # slice
        ds = ds.isel(time_counter=slice(offset,None,slice_dist))

        # save
        with ProgressBar():
            out_fn = self.proc_preamble + file_id_slice + grid_str + '.nc'
            ds.to_netcdf(out_fn)

file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_15mi_20121209_20121211_'
file_id_slice = 'SOCHIC_PATCH_30mi_20121209_20121211_'

dws = downsample('TRD00', file_id)
#dws.downsample_time('grid_U', file_id_slice, slice_dist=4) 
#dws.downsample_time('grid_T', file_id_slice, slice_dist=4) 
#dws.downsample_time('grid_V', file_id_slice, slice_dist=4) 
#dws.downsample_time('grid_W', file_id_slice, slice_dist=4) 
#dws.downsample_time('icemod', file_id_slice, slice_dist=4) 
dws.downsample_time('rhoW', file_id_slice, slice_dist=2) 
