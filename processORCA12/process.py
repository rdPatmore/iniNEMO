import xarray as xr
import numpy as np
import gridding
import datetime
#import dask

#dask.config.set(scheduler='single-threaded')

def process(pos='T', year='2014', month='01', day='', t0=False, opendap=False):

    # get subset coordinate indices for scalar isel
    coords = xr.open_dataset('DataIn/coordinates.nc', decode_times=False)
    xlen = coords.sizes['x']
    ylen = coords.sizes['y']
    coords = coords.assign({'i': (('x'), np.arange(xlen))})
    coords = coords.assign({'j': (('y'), np.arange(ylen))})

    coord_subset = cut(coords)
    lon0 = int(coord_subset.i.min())
    lon1 = int(coord_subset.i.max()+1)
    lat0 = int(coord_subset.j.min())
    lat1 = int(coord_subset.j.max()+1)

    #paths = ('http://opendap4gws.jasmin.ac.uk/thredds/nemo/dodsC/nemo_clim_agg/'
    #         'ORCA0083-N06_grid_T_' + year)
    date =  year + month
    if opendap:
        indir = ('http://opendap4gws.jasmin.ac.uk/thredds/nemo/dodsC/grid_T/'
                  + year + '/ORCA0083-N06_')
    else:
        indir = 'DataIn/ORCA0083-N06_'

    if day != '':
        days = [day]
    else:
        if year == '2014':
            days = ['01','06','11','16','21','26','31']
        if year == '2015':
            if month == '01':
                days = ['05','10','15','20','25','30']
            if month == '11':
                days = ['06','11','16','21','26']

    paths = []
    for d in days:
        paths.append(indir + date + d + 'd05' + pos + '.nc')

    outdir = 'DataOut/'

    if pos == 'T':
        chunk={'deptht':1}#, 'x':10, 'y':10}
        drop = ['tos', 'tossq','sos','zossq', 'taum','mldkz5','mldr10_1']
        
       #ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05T.nc',
       #                       mask_and_scale=False, chunks=chunk,
       #                       drop_variables=drop, combine='by_coords',
       #                       decode_cf=False
       #                       )
        kwargs = {'mask_and_scale': False, 'chunks': chunk,
                  'drop_variables': drop, 'decode_cf': False,
                  'decode_times': False}
        ds = xr.open_mfdataset(paths, **kwargs)

        print ('loading')
        #time_period = year + '-' + month# + '-01'
        #ds = ds.isel(x=slice(3380,3515), y=slice(450,720)).load()
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1)).load()
        print ('loaded')
        #ds = ds.sel(time_counter=time_period).load()
        #print ('sel month')
        #for var in ds.data_vars:
        #    ds = ds.assign_coords(time_counter=ds.time_centered)
        print ('time_centered', ds.time_centered.values)
        #print ('time_centered_bounds', ds.time_centered_bounds.values)
        print ('time_counter', ds.time_counter.values)
        #print ('time_counter_bounds', ds.time_counter_bounds.values)

            #print ('loading')
            #ds = ds.load()
            #print ('done')

        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds','nav_lat','nav_lon'])

        # conform names
        ds = ds.rename({'so':    'vosaline',
                        'thetao':'votemper',
                        'zos'   :'sossheig',
                        'wfo'   :'sowaflup',
                        'rsntds':'soshfldo',
                        'tohfls':'sohefldo'})

        #ds['vosaline'] = ds.vosaline.fillna(40)
        #ds['votemper'] = ds.votemper.fillna(40)
        #ds['vosaline'] = xr.where(ds.vosaline > 40, 40, ds.vosaline)
        #ds['votemper'] = xr.where(ds.votemper > 40, 40, ds.votemper)

        #for var in ['sossheig', 'sowaflup', 'soshfldo', 'sohefldo']:
        #    ds[var] = ds[var].fillna(0)
            #ds[var] = xr.where(ds[var] > 40,   0, ds[var])
        
    if pos == 'U':
        drop = ['tauuo','uos']
        chunk={'depthu':1}
        ds = xr.open_mfdataset(paths,
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False
                               )#.isel(x=slice(3400,3500), y=slice(500,700))
        print ('loading')
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1)).load()
        print ('done')

        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds','nav_lon','nav_lat'])
        ds = ds.rename({'uo':'vozocrtx'})
        #ds['vozocrtx'] = ds.vozocrtx.fillna(0.0)
        #ds['vozocrtx'] = xr.where(ds.vozocrtx > 40, 0, ds.vozocrtx)
       
        #ds = ds.vozocrtx.to_dataset()#.load()
        #ds = ds.drop(['time_centered','time_counter']).load()

    if pos == 'V':
        drop = ['tauvo','vos']
        chunk={'depthv':1}
        ds = xr.open_mfdataset(paths,
                               mask_and_scale=False, chunks=chunk,
                               drop_variables=drop, combine='by_coords',
                               decode_cf=False
                               )
        print ('loading')
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1)).load()
        print ('done')

        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds','nav_lon','nav_lat'])

        ds = ds.rename({'vo':'vomecrty'})
        #ds['vomecrty'] = ds.vomecrty.fillna(0.0)
        #ds['vomecrty'] = xr.where(ds.vomecrty > 40, 0, ds.vomecrty)

    if pos == 'I':
        drop = ['snowpre', 'sip', 'ist_ipa', 'uice_ipa', 'vice_ipa', 
                'utau_ice', 'vtau_ice', 'qsr_io_cea', 'qns_io_cea']
        ds = xr.open_mfdataset(paths, drop_variables=drop,
                               mask_and_scale=False, combine='by_coords',
                               decode_cf=False
                               )
        print ('loading')
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1)).load()
        print ('done')

        ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds','longitude','latitude'])
        #                    'time_centered_bounds','nav_lon','nav_lat'])
        #ds = ds.drop_dims('axis_nbounds')
      

        ds = ds.rename({'ice_pres':'siconc',
                        'sit': 'sithic',
                        'snd': 'snthic'})
        for var in ['siconc','sithic','snthic']:
            ds[var] = ds[var].fillna(0.0)
            # boundary hack for perio issues
            ds[var][:,-1] = np.nan
            ds[var][:,0] = np.nan
            ds[var][:,:,-1] = np.nan
            ds[var][:,:,0] = np.nan

    if pos == 'Tsurf':
        ds = xr.open_dataset(indir + 'ORCA0083-N06_20150105d05T.nc',
                             mask_and_scale=False)
        ds = ds.rename({'zos':   'sossheig'})
        ds['sossheig']  = xr.where(ds.sossheig == np.nan,  0, ds.sossheig)
        ds['sossheig']  = xr.where(ds.sossheig > 40,       0, ds.sossheig)
        ds = ds.sossheig.to_dataset()
        #ds = ds.drop(['time_counter','time_centered'])

    #ds.attrs['DOMAIN_number_total'] = 1
    #ds.attrs['nj'] = ds.attrs['DOMAIN_size_global'][1]

#    ds.attrs['ni'] = int(len(ds.x))
#    ds.attrs['nj'] = int(len(ds.y))
#    ds.attrs['DOMAIN_number_total'] = 1
#    ds.attrs['DOMAIN_size_global'] = [ds.attrs['ni'],ds.attrs['nj']]
#    ds.attrs['periodicity'] = 0
    ds.attrs = {}
    #ds.siconc.attrs['periodicity'] = 0
    #ds.sithic.attrs['periodicity'] = 0
    #ds.snthic.attrs['periodicity'] = 0
    #ds['jperio'] = 0

    print (ds)
    ds.time_counter.encoding['dtype'] = np.float64
    ##ds['time_counter'] = (ds.time_counter - 
    ##                      datetime.timedelta(days=2,hours=12).total_seconds())
    ##ds['time_centered'] = (ds.time_centered - 
    ##                      datetime.timedelta(days=2,hours=12).total_seconds())
    #ds.time_counter.attrs['units'] = 'seconds since 1900-01-01'
    #ds.time_centered.attrs['units'] = 'seconds since 1900-01-01'
    #ds = xr.decode_cf(ds)

    if t0: day='d' + day
    date_srt = 'y' + year + 'm' + month + day + '_'
    comp = dict(zlib=True, complevel=6)
    encoding = {var: comp for var in ds.data_vars}
    #if t0: # save first time step for restarts
    #     ds = ds.isel(time_counter=0)
    #     ds.to_netcdf(
    #          outdir + 'ORCA0083-N06_' + date_srt + pos + '_conform.nc',
    #                  encoding=encoding, unlimited_dims='time_counter')
    ds.to_netcdf(outdir + 'ORCA0083-N06_' + date_srt + pos + '_conform.nc',
                     encoding=encoding, unlimited_dims='time_counter')

def cut(ds):
    ds = ds.where((ds.nav_lon > -5)  & (ds.nav_lon < 5) & 
                  (ds.nav_lat > -65) & (ds.nav_lat < -55), drop=True)
    return ds

def subset_coords():
    indir  = 'DataIn/'
    outdir = 'DataOut/'

    drop = ['nav_lev','time_steps']
    ds = xr.open_dataset(indir + 'coordinates.nc', decode_cf=False, 
                         drop_variables=drop)#.isel(
                     #   x=slice(3400,3500), y=slice(500,700))
    ds = cut(ds)
    ds = ds.squeeze('time')
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'coordinates_subset.nc', encoding=encoding)

def set_halo(ds, field, fill_value=0):
    ''' add halo to field '''

    print (ds)
    ds[field].loc[{'x':0}] = fill_value
    ds[field].loc[{'x':-1}] = fill_value
    ds[field].loc[{'y':0}] = fill_value
    ds[field].loc[{'y':-1}] = fill_value

    return ds

def subset_bathy():
    ''' subset bathy to 8 degree patch of weddell sea'''

    outdir = 'DataOut/'
    path = ('https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/dodsC/'
           +'meomopendap/extract/ORCA12.L46/ORCA12.L46-I/'
           +'bathymetry_ORCA12_V3.3.nc')
    ds = xr.open_dataset(path, decode_cf=False)

    ds = cut(ds)
    #ds = set_halo(ds, 'Bathymetry', 0)
    #ds = set_halo(ds, 'mask', 0)

    ds.Bathymetry.attrs['_FillValue']=0
    ds.mask.attrs['_FillValue']=0

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'bathy_8deg.nc', encoding=encoding)

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

   if pos == 'U':
       chunk = {'depthu':1}
       var_keys = ['vozocrtx']

   if pos == 'V':
       chunk = {'depthv':1}
       var_keys = ['vomecrty']

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

for pos in ['I']:
    process(pos=pos, year='2015', month='11', day='06', opendap=False, t0=True)
    process(pos=pos, year='2015', month='11', opendap=False, t0=False)
#process(pos='U', year='2015', month='01', day='10', opendap=False, t0=True)
#process(pos='V', year='2015', month='01', day='10', opendap=False, t0=True)
#process(pos='T', year='2015', month='01', opendap=False)
#process(pos='V', year='2014', month='12', opendap=False)
#subset_bathy()
#cut_orca('U')
#cut_orca('V')
