import xarray as xr
import numpy as np


def process(pos='T'):

    indir  = 'DataIn/'
    outdir = 'DataOut/'

    cfg = xr.open_dataset(indir + 'domain_cfg.nc')

    if pos == 'T':
        ds = xr.open_dataset(indir + 'ORCA0083-N06_20150105d05T.nc',
                             mask_and_scale=False)
        ds = ds.rename({'so':    'vosaline',
                        'thetao':'votemper',
                        'zos'   :'sossheig'})
        salt = xr.where(ds.vosaline == np.nan, 40, ds.vosaline)
        temp = xr.where(ds.votemper == np.nan, 40, ds.votemper)
        ssh  = xr.where(ds.sossheig == np.nan,  0, ds.sossheig)
        salt = xr.where(salt > 40, 40, ds.vosaline)
        temp = xr.where(temp > 40, 40, ds.votemper)
        ssh  = xr.where(ssh > 40,   0, ds.sossheig)
        ds = xr.merge([salt, temp, ssh])
        ds = ds.drop(['time_counter','time_centered'])

    if pos == 'U':
        ds = xr.open_dataset(indir + 'ORCA0083-N06_20150105d05U.nc',
                             mask_and_scale=False)
        ds = ds.rename({'uo':'vozocrtx'})
        ds['vozocrtx'] = xr.where(ds.vozocrtx == np.nan, 0, ds.vozocrtx)
        ds['vozocrtx'] = xr.where(ds.vozocrtx > 40, 0, ds.vozocrtx)
        ds = ds.vozocrtx.to_dataset()
        ds = ds.drop(['time_counter','time_centered'])

    if pos == 'V':
        ds = xr.open_dataset(indir + 'ORCA0083-N06_20150105d05V.nc',
                             mask_and_scale=False)
        ds = ds.rename({'vo':'vomecrty'})
        ds['vomecrty'] = xr.where(ds.vomecrty == np.nan, 0, ds.vomecrty)
        ds['vomecrty'] = xr.where(ds.vomecrty > 40, 0, ds.vomecrty)
        ds = ds.vomecrty.to_dataset()
        ds = ds.drop(['time_counter','time_centered'])

    if pos == 'Tsurf':
        ds = xr.open_dataset(indir + 'ORCA0083-N06_20150105d05T.nc',
                             mask_and_scale=False)
        ds = ds.rename({'zos':   'sossheig'})
        ds['sossheig']  = xr.where(ds.sossheig == np.nan,  0, ds.sossheig)
        ds['sossheig']  = xr.where(ds.sossheig > 40,       0, ds.sossheig)
        ds = ds.sossheig.to_dataset()
        ds = ds.drop(['time_counter','time_centered'])

    
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'ORCA0083-N06_20150105d05' + pos + '_conform.nc',
                 encoding=encoding, unlimited_dims='time_counter')

process('T')
