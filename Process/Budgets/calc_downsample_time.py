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

    def cut_length(self, grid_str, file_id_slice, 
                   start_date=None, end_date=None):
        ''' reduce start and end and save '''

        # get data
        kwargs = {'chunks':'auto' ,'decode_cf':True} 
        fn = self.raw_preamble + self.file_id + grid_str + '.nc'
        ds = xr.open_dataset(fn, **kwargs)

        # trim
        ds = ds.sel(time_counter=slice(start_date,end_date))

        print (ds)
        # save
        with ProgressBar():
            out_fn = self.raw_preamble + file_id_slice + grid_str + '.nc'
            ds.to_netcdf(out_fn)

if __name__ == '__main__':
    file_id = 'SOCHIC_PATCH_30mi_20121223_20121226_'
    file_id_slice = 'SOCHIC_PATCH_30mi_20121224_20121225_'
    start = '2012-12-24 12:00:00'
    end = '2012-12-25 12:00:00'
    
    dws = downsample('TRD02', file_id)
    grid_list = ['grid_T','grid_U','grid_V','grid_W','icemod','momu','momv']
    for grid in grid_list:
        dws.cut_length(grid, file_id_slice, start, end) 
    #dws.downsample_time('grid_T', file_id_slice, slice_dist=4) 
    #dws.downsample_time('grid_V', file_id_slice, slice_dist=4) 
    #dws.downsample_time('grid_W', file_id_slice, slice_dist=4) 
    #dws.downsample_time('icemod', file_id_slice, slice_dist=4) 
    #dws.downsample_time('rhoW', file_id_slice, slice_dist=2) 
