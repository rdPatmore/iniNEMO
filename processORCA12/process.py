import xarray as xr
import numpy as np
import gridding
import datetime
#import dask

#dask.config.set(scheduler='single-threaded')

def process(pos='T', year='2014', month='', day='', t0=False, opendap=False):

    # get subset coordinate indices for scalar isel
    coords = xr.open_dataset('../SourceData/coordinates.nc', decode_times=False)
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
        indir = ('http://opendap4gws.jasmin.ac.uk/thredds/nemo/dodsC/grid_' + 
                  pos + '/' + year + '/ORCA0083-N06_')
    else:
        indir = '../SourceData/ORCA0083-N06_'

    #if day != '':
    #    days = [day]
    #else:
    #    if year == '2014':
    #        days = ['01','06','11','16','21','26','31']
    #    if year == '2015':
    #        if month == '01':
    #            days = ['05','10','15','20','25','30']
    #        if month == '11':
    #            days = ['06','11','16','21','26']

    if not opendap:
        #if (month and day) == 0:
        print ('oui oui')
        if month == '':
            paths = indir +  year + '*d05' + pos + '.nc'
        elif day == '':
            paths = indir +  year + month + '*d05' + pos + '.nc'
        else:
            paths = indir +  year + month + day + 'd05' + pos + '.nc'
        #else:
        #    paths = []
        #    for d in days:
        #        paths.append(indir + date + d + 'd05' + pos + '.nc')

    outdir = '../DataOut/'

    if pos == 'T':
        chunk={'deptht':1}#, 'x':10, 'y':10}
        drop = ['tos', 'tossq','sos','zossq', 'taum','mldkz5','mldr10_1',
                'sst','sss','sosflxdo','sowindsp','soprecip','e3t']  
        
       #ds = xr.open_mfdataset(indir + 'ORCA0083-N06_201501*d05T.nc',
       #                       mask_and_scale=False, chunks=chunk,
       #                       drop_variables=drop, combine='by_coords',
       #                       decode_cf=False
       #                       )
        kwargs = {'mask_and_scale': False, 'chunks': chunk,
                  'drop_variables': drop, 'decode_cf': False,
                  'decode_times': False}
        if opendap:
            ds = xr.open_dataset(path, **kwargs)
        else:
            ds = xr.open_mfdataset(paths, **kwargs)

        print ('loading')
        #time_period = year + '-' + month# + '-01'
        #ds = ds.isel(x=slice(3380,3515), y=slice(450,720)).load()
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1))#.load()
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
        try:
            ds = ds.rename({'so':    'vosaline',
                            'thetao':'votemper',
                            'zos'   :'sossheig',
                            'wfo'   :'sowaflup',
                            'rsntds':'soshfldo',
                            'tohfls':'sohefldo'})
        except:
            ds = ds.rename({'salin': 'vosaline',
                            'potemp':'votemper',
                            'ssh'   :'sossheig',
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
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1))#.load()
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
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1))#.load()
        print ('done')

        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds','nav_lon','nav_lat'])

        ds = ds.rename({'vo':'vomecrty'})
        #ds['vomecrty'] = ds.vomecrty.fillna(0.0)
        #ds['vomecrty'] = xr.where(ds.vomecrty > 40, 0, ds.vomecrty)

    if pos == 'I':
        drop = ['snowpre', 'sip', 'utau_ice', 'vtau_ice',
                'qsr_io_cea', 'qns_io_cea']
        ds = xr.open_mfdataset(paths, drop_variables=drop,
                               mask_and_scale=False, combine='by_coords',
                               decode_cf=False
                               )
        print ('loading')
        ds = ds.isel(x=slice(lon0,lon1), y=slice(lat0,lat1))#.load()
        print ('done')

        #ds = ds.rename({'nav_lon': 'longitude', 'nav_lat': 'latitude'})
        ds = ds.set_coords(['time_counter_bounds','time_centered',
                            'time_centered_bounds','nav_lon','nav_lat'])
        #                    'time_centered_bounds','longitude','latitude'])
        #ds = ds.drop_dims('axis_nbounds')
      

        ds = ds.rename({'ice_pres':'siconc',
                        'sit': 'sithic',
                        'snd': 'snthic',
                        'ist_ipa':'sitemp',
                        'uice_ipa':'u_ice',
                        'vice_ipa':'v_ice'})

       
        ice_shape = (ds.time_counter.shape[0], ds.y.shape[0], ds.x.shape[0])
        print (ice_shape)
        ds = ds.assign({'sisalt': (('time_counter', 'y', 'x'), 
                        10 * np.ones(ice_shape))})

        for var in ['siconc','sithic','snthic','v_ice', 'u_ice', 'sitemp']:
            ds[var] = ds[var].fillna(0.0)
        for var in ['siconc','sithic','snthic','v_ice', 'u_ice', 'sitemp',
                    'sisalt']:
            # boundary hack for perio issues
            ds = set_halo(ds, var, fill_value=np.nan)
            #ds[var][:,-1] = np.nan
            #ds[var][:,0] = np.nan
            #ds[var][:,:,-1] = np.nan
            #ds[var][:,:,0] = np.nan

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

    #if t0: day='d' + day
    date_srt = year + month + day + '_'
    comp = dict(zlib=True, complevel=6)
    encoding = {var: comp for var in ds.data_vars}
    #if t0: # save first time step for restarts
    #     ds = ds.isel(time_counter=0)
    #     ds.to_netcdf(
    #          outdir + 'ORCA0083-N06_' + date_srt + pos + '_conform.nc',
    #                  encoding=encoding, unlimited_dims='time_counter')
    ds.to_netcdf(outdir + 'ORCA0083-N06_' + date_srt + pos + '_conform.nc',
                     encoding=encoding, unlimited_dims='time_counter')

def merge_year(year, pos):
    ''' merge NOC orca12 data into one file in prep for NESTING module '''
    
    path = '../SourceData/ORCA0083-N06_' + str(year) + '*d05' + pos + '.nc'
    ds = xr.open_mfdataset(path, chunks={'time_counter':1}, decode_cf=False)
    ds.to_netcdf('../ORCA12_' + str(year) + '_' + pos + '.nc', 
                  unlimited_dims='time_counter')

def cut(ds):
    ds = ds.where((ds.nav_lon > -5)  & (ds.nav_lon < 5) & 
                  (ds.nav_lat > -65) & (ds.nav_lat < -55), drop=True)
    return ds

def subset_coords():
    indir  = '/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_12/'
    outdir = '../OrcaCutData/'

    drop = ['nav_lev','time_steps']
    ds = xr.open_dataset(indir + 'coordinates.nc', decode_cf=False, 
                         drop_variables=drop)#.isel(
                     #   x=slice(3400,3500), y=slice(500,700))
    ds = cut(ds)
    #ds = ds.squeeze('time')
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'coordinates_subset.nc', encoding=encoding)

def set_halo(ds, field, method='where', fill_value=0):
    ''' add halo to field '''

    print (ds)
    if method == 'where':
        xend = ds.x[-1]
        yend = ds.y[-1]
        ds[field] = xr.where(ds.x == 0, fill_value, ds[field])
        ds[field] = xr.where(ds.x == xend, fill_value, ds[field])
        ds[field] = xr.where(ds.y == 0, fill_value, ds[field])
        ds[field] = xr.where(ds.y == yend, fill_value, ds[field])
        ds[field] = ds[field].transpose('time_counter','y','x')
    else:
        ds[field].loc[{'x':0}] = fill_value
        ds[field].loc[{'x':-1}] = fill_value
        ds[field].loc[{'y':0}] = fill_value
        ds[field].loc[{'y':-1}] = fill_value
    print (ds)

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

def subset_zmesh():
    ''' subset zmesh to 8 degree patch of weddell sea'''

    outdir = 'DataOut/'
    path = ('https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/dodsC/'
           +'meomopendap/extract/ORCA12.L46/ORCA12.L46-I/'
           +'mesh_zgr.nc')
    ds = xr.open_dataset(path, decode_cf=False)

    ds = cut(ds)
    
    ds.mbathy.attrs['_FillValue']=0

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(outdir + 'mesh_zgr_8deg.nc', encoding=encoding)

def cut_orca(pos, year=0, month=0):
   '''
   regrid orca to chosen model grid
   pos can be in {T,U,V}
   '''

   indir  = '../SourceData/'
   outdir = '../OrcaCutData/'
   
   #date = 'y' + str(year) + 'm' + str(month)

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

   if pos == 'I':
       chunk = None
       var_keys = ['siconc', 'sithic', 'snthic','sitemp','u_ice','v_ice']

   coord = xr.open_dataset(outdir + 'coordinates_subset.nc',
                           decode_times=False)
   ds    = xr.open_dataset(
            '../DataOut/ORCA0083-N06_' +  year + '_' + pos + '_conform.nc',
                           mask_and_scale=False, decode_cf=False)
   
   coord = coord.drop('time')

   arrs = []
   for var in var_keys: 
       arrs.append(gridding.regrid(ds, coord, var))
   print (arrs[0])
   ds_cut = xr.merge(arrs)
   ds_cut['nav_lat'] = coord.nav_lat
   ds_cut['nav_lon'] = coord.nav_lon
   ds_cut['time_counter'] = ds.time_counter
   ds_cut['time_counter_bounds'] = ds.time_counter_bounds
   ds_cut['time_centered'] = ds.time_centered
   ds_cut['time_centered_bounds'] = ds.time_centered_bounds

   comp = dict(zlib=True, complevel=9)
   encoding = {var: comp for var in ds_cut.data_vars}
   print (encoding)
   ds_cut.to_netcdf(outdir + 'ORCA_PATCH_' + year + '_' + pos + '.nc',
                encoding=encoding)


for pos in ['T','U','V','I']:
    process(pos=pos, year='2014', opendap=False)
#    process(pos=pos, year='2013', month='01', day='05', opendap=False)
    #process(pos=pos, year='2015', month='11', day='06', opendap=True, t0=True)
#process(pos='U', year='2015', month='01', day='10', opendap=False, t0=True)
#process(pos='V', year='2015', month='01', day='10', opendap=False, t0=True)
#process(pos='T', year='2015', month='01', opendap=False)
#process(pos='V', year='2014', month='12', opendap=False)
#subset_bathy()
#cut_orca('I', year='2012')
#merge_year(2012, 'T')
