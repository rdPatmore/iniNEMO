# convert a bunch of netcdf files to zarr

import numpy as np
import xarray as xr
import pandas as pd
import numcodecs
import config

def nc_to_zarr(case):

    #dates = pd.date_range(start='2017-01-01 00:00',
    #                      end='2017-12-31 23:00', freq='1h')
    
    appends = ['_grid_T', '_grid_U', '_grid_V', '_grid_W','_icemod'][::-1]
    dates = '20120101_20121231'
    # construct list of netcdf files
    path = config.data_path() + case + '/'
    files = [path + 'SOCHIC_PATCH_24h_' + dates + append + '.nc' for append 
                                                                  in appends]
    
    for f in files:
        ds = xr.open_dataset(f)#, chunks={'time_counter':1})
    
        numcodecs.blosc.use_threads = False
    
        new_fname = f.rstrip('.nc') + '.zarr'
        print (new_fname)
        ds.to_zarr(new_fname, mode='w', consolidated=True)

def open_zarr(case, append):
    dates = '20120101_20121231'
    ds = xr.open_zarr(config.data_path() + case + '/' + 
                         'SOCHIC_PATCH_24h_' + dates + append +
                         '.zarr')#,  engine='zarr')
    print (ds)
    
    
    
nc_to_zarr('EXP10')
