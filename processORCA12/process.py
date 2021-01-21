import xarray as xr
import numpy as np


def process(pos='T'):

    indir  = 'DataIn/'
    outdir = 'DataOut/'

    if pos == 'T':
        chunk={'deptht':1}
        drop = ['tos', 'tossq','sos','zossq','wfo','rsntds','tohfls',
                'taum','mldkz5','mldr10_1']
        ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05T.nc',
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False
                               ).isel(x=slice(3400,3500), y=slice(500,700))
        ds = ds.rename({'so':    'vosaline',
                        'thetao':'votemper',
                        'zos'   :'sossheig'})
        ds['vosaline'] = ds.vosaline.fillna(40)
        ds['votemper'] = ds.votemper.fillna(40)
        ds['sossheig'] = ds.sossheig.fillna(0)
        ds['vosaline'] = xr.where(ds.vosaline > 40, 40, ds.vosaline)
        ds['votemper'] = xr.where(ds.votemper > 40, 40, ds.votemper)
        ds['sossheig'] = xr.where(ds.sossheig > 40,   0, ds.sossheig)

    if pos == 'U':
        drop = ['tauuo','uos']
        chunk={'depthu':1}
        ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05U.nc',
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False,
                               ).isel(x=slice(3400,3500), y=slice(500,700))
        ds = ds.rename({'uo':'vozocrtx'})
        ds['vozocrtx'] = ds.vozocrtx.fillna(0.0)
        ds['vozocrtx'] = xr.where(ds.vozocrtx > 40, 0, ds.vozocrtx)
       
        #ds = ds.vozocrtx.to_dataset()#.load()
        #ds = ds.drop(['time_centered','time_counter']).load()

    if pos == 'V':
        drop = ['tauvo','vos']
        chunk={'depthv':1}
        ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05V.nc',
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False
                               ).isel(x=slice(3400,3500), y=slice(500,700))
        ds = ds.rename({'vo':'vomecrty'})
        ds['vomecrty'] = ds.vomecrty.fillna(0.0)
        ds['vomecrty'] = xr.where(ds.vomecrty > 40, 0, ds.vomecrty)

    if pos == 'Tsurf':
        ds = xr.open_dataset(indir + 'ORCA0083-N06_20150105d05T.nc',
                             mask_and_scale=False)
        ds = ds.rename({'zos':   'sossheig'})
        ds['sossheig']  = xr.where(ds.sossheig == np.nan,  0, ds.sossheig)
        ds['sossheig']  = xr.where(ds.sossheig > 40,       0, ds.sossheig)
        ds = ds.sossheig.to_dataset()
        #ds = ds.drop(['time_counter','time_centered'])

    ds.attrs['DOMAIN_number_total'] = 1
    ds.attrs['nj'] = ds.attrs['DOMAIN_size_global'][1]
    ds = ds.load()

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'ORCA0083-N06_' + pos + '_conform.nc',
                 encoding=encoding, unlimited_dims='time_counter')

def subset_coords():
    indir  = 'DataIn/'
    outdir = 'DataOut/'

    ds = xr.open_dataset(indir + 'coordinates.nc', decode_cf=False).isel(
                        x=slice(3400,3500), y=slice(500,700))
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'coordinates_subset.nc', encoding=encoding)

process('T')
process('U')
process('V')
