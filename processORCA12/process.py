import xarray as xr
import numpy as np
import gridding


def process(pos='T'):

    indir  = 'DataIn/'
    outdir = 'DataOut/'

    if pos == 'T':
        chunk={'deptht':1}
        drop = ['tos', 'tossq','sos','zossq', 'taum','mldkz5','mldr10_1']
        ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05T.nc',
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False
                               )#.isel(x=slice(3400,3500), y=slice(500,700))
        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds'])
        ds = cut(ds)

        ## reverse water flux
        #ds['wfo'] = - ds.wfo

        # conform names
        ds = ds.rename({'so':    'vosaline',
                        'thetao':'votemper',
                        'zos'   :'sossheig',
                        'wfo'   :'sowaflup',
                        'rsntds':'soshfldo',
                        'tohfls':'sohefldo'})

        ds['vosaline'] = ds.vosaline.fillna(40)
        ds['votemper'] = ds.votemper.fillna(40)
        ds['vosaline'] = xr.where(ds.vosaline > 40, 40, ds.vosaline)
        ds['votemper'] = xr.where(ds.votemper > 40, 40, ds.votemper)

        for var in ['sossheig', 'sowaflup', 'soshfldo', 'sohefldo']:
            ds[var] = ds[var].fillna(0)
            #ds[var] = xr.where(ds[var] > 40,   0, ds[var])
        
    if pos == 'U':
        drop = ['tauuo','uos']
        chunk={'depthu':1}
        ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05U.nc',
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False,
                               )#.isel(x=slice(3400,3500), y=slice(500,700))
        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds'])
        ds = cut(ds)
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
                               )#.isel(x=slice(3400,3500), y=slice(500,700))
        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds'])
        ds = cut(ds)
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
    # save first time step for restarts
    ds = ds.isel(time_counter=0)
    ds.to_netcdf(outdir + 'ORCA0083-N06_' + pos + '_ts0_conform.nc',
                 encoding=encoding, unlimited_dims='time_counter')

def cut(ds):
    ds = ds.where(ds.nav_lon < 5, drop=True)
    ds = ds.where(ds.nav_lon > -5, drop=True)
    ds = ds.where(ds.nav_lat < -55, drop=True)
    ds = ds.where(ds.nav_lat > -65, drop=True)
    ds['nav_lon'] = ds.nav_lon.isel(time_counter=0)
    ds['nav_lat'] = ds.nav_lat.isel(time_counter=0)
   
    ds.attrs['ni'] = int(len(ds.x))
    ds.attrs['nj'] = int(len(ds.y))
    ds.attrs['DOMAIN_number_total'] = 1
    ds.attrs['DOMAIN_size_global'] = [ds.attrs['ni'],ds.attrs['nj']]
    return ds

def cut_ds(ds):
    ds = ds.where(ds.nav_lon < 5, drop=True)
    ds = ds.where(ds.nav_lon > -5, drop=True)
    ds = ds.where(ds.nav_lat < -55, drop=True)
    ds = ds.where(ds.nav_lat > -65, drop=True)
    return ds

def subset_coords():
    indir  = 'DataIn/'
    outdir = 'DataOut/'

    drop = ['nav_lev','time_steps']
    ds = xr.open_dataset(indir + 'coordinates.nc', decode_cf=False, 
                         drop_variables=drop)#.isel(
                     #   x=slice(3400,3500), y=slice(500,700))
    ds = cut_ds(ds)
    ds = ds.squeeze('time')
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'coordinates_subset.nc', encoding=encoding)

def cut_orca(pos):
   '''
   regrid orca to chosen model grid
   pos can be in {T,U,V}
   '''

   indir  = 'DataIn/'
   outdir = 'DataOut/'
   
   if pos == 'T':
       chunk = {'deptht':1}
       var_keys = ['votemper', 'vosaline', 'sossheig',
                   'sowaflup', 'soshfldo', 'sohefldo']

   coord = xr.open_dataset('../SourceData/coordinates.nc', decode_times=False)
   ds    = xr.open_dataset(outdir + 'ORCA0083-N06_' + pos + '_conform.nc',
                           mask_and_scale=False, decode_cf=False)
   
   coord = coord.drop('time')

   for var in var_keys: 
       ds = gridding.regrid(ds, coord, var)
   ds['nav_lat'] = coord.nav_lat
   ds['nav_lon'] = coord.nav_lon
   ds['time_counter'] = ds.time_counter

   comp = dict(zlib=True, complevel=9)
   encoding = {var: comp for var in ds.data_vars}
   ds.to_netcdf(outdir + 'ORCA_PATCH_' + pos + '.nc', encoding=encoding)

process(pos='T')
#process(pos='U')
#process(pos='V')
#subset_coords()
cut_orca('T')
