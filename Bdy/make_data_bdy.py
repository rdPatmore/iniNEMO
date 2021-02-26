import xarray as xr
import numpy  as np

def get_side(data, side, pos, offset=0):
    vel_block=0
    print ('SIDE: ', side)
    mesh_mask = xr.open_dataset('../SourceData/mesh_mask.nc').rename({
        'x':'X', 'y':'Y'}).squeeze('time_counter').reset_coords('time_counter')
    if side == 'west':
                           #Y=slice(2+offset,-1-offset)
        arrayX = data.isel(X=0 + offset, 
                          Y=slice(offset,int(data.attrs['bdy_end'])-1-offset)
                          ).reset_coords('X', drop=True)
        mesh_mask = mesh_mask.isel(X=1 + offset,
                    Y=slice(1+offset,int(arrayX.attrs['bdy_end'] - offset)))
        dim='Y'
        bdy_pos = int(arrayX.attrs['bdy_deb'] + offset)
    if side == 'east':
        if pos == 'U':
            vel_block=1 
        arrayX = data.isel(X=0 + offset + vel_block,
                          Y=slice(offset,int(data.attrs['bdy_end'])-1-offset)
                           ).reset_coords('X', drop=True)  
        mesh_mask = mesh_mask.isel(X=-2 - offset - vel_block,
                    Y=slice(1+ offset,int(arrayX.attrs['bdy_end'])-offset))
        bdy_pos = int(arrayX.attrs['bdy_ind']+ 1 - offset - vel_block)
        dim='Y'
    if side == 'south':
        arrayX = data.isel(Y=0 + offset,
                           X=slice(offset,int(data.attrs['bdy_end'])-1-offset)
                          ).reset_coords('Y', drop=True)
        mesh_mask = mesh_mask.isel(Y=1 + offset,
                    X=slice(1+offset,int(arrayX.attrs['bdy_end'])-offset))
        dim='X'
        bdy_pos = int(arrayX.attrs['bdy_deb'] + offset)
    if side == 'north':
        if pos == 'V':
            vel_block=1
        arrayX = data.isel(Y=0 + offset + vel_block,
                           X=slice(offset,int(data.attrs['bdy_end'])-1-offset)
                           ).reset_coords('Y', drop=True)
        #print (mesh_mask.isel(Y=-2-offset-vel_block).nav_lon)
        mesh_mask = mesh_mask.isel(Y=-2 - offset - vel_block,
                    X=slice(1+offset,int(arrayX.attrs['bdy_end'])-offset))
        bdy_pos = int(arrayX.attrs['bdy_ind'] + 1 - offset - vel_block)
        #print (mesh_mask.nav_lon)
        ##print (arrayX.nav_lon)
        #print (mesh_mask.nav_lat)
        #print (arrayX.nav_lat)
        #print ((mesh_mask.nav_lon - arrayX.nav_lon).round(4))
        #print ((mesh_mask.nav_lat - arrayX.nav_lat).round(4))
        dim='X'
    if pos == 'T':
        arrayX['gdept'] = mesh_mask['gdept_0']
        arrayX['e3t'] = mesh_mask['e3t_0']
        if side in ['north', 'south']:
            nba = 'nbit'
            nbb = 'nbjt'
        if side in ['east', 'west']:
            nba = 'nbjt'
            nbb = 'nbit'
        if side in ['north']:
            arrayX = arrayX.sortby('X', ascending=False)
        if side in ['west']:
            arrayX = arrayX.sortby('Y', ascending=False)
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbt'], arrayX[dim].values + 1),
                         nbb: (['xbt'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbrt': (['xbt'], np.full(arrayX[dim].shape, 1 + offset)),
                         'deptht':(['deptht'], arrayX.deptht.values),
                         'nav_lon':(['xbt'], arrayX.nav_lon.values),
                         'nav_lat':(['xbt'], arrayX.nav_lat.values),
                 'gdept':(['deptht','xbt'], arrayX.gdept.values),
                     'e3t':(['deptht','xbt'], arrayX.e3t.values),
              'sossheig':  (['time_counter','xbt'], arrayX.sossheig.values),
         'vosaline':  (['time_counter','deptht','xbt'], arrayX.vosaline.values),
         'votemper':  (['time_counter','deptht','xbt'], arrayX.votemper.values)}
                       ).expand_dims('yb')
        ds.deptht.attrs['long_name']='Vertical T levels'
        ds.deptht.attrs['units']='m'
        ds.vosaline.attrs['long_name']='Salinity'
        ds.vosaline.attrs['units']='psu'
        ds.votemper.attrs['long_name']='Temperature'
        ds.votemper.attrs['units']='degC'
        ds.nbrt.attrs['long_name']='bdy discrete distance'
        ds.nbrt.attrs['units']='unitless'
        ds.nbjt.attrs['long_name']='bdy j index'
        ds.nbjt.attrs['units']='unitless'
        ds.nbit.attrs['long_name']='bdy i index'
        ds.nbit.attrs['units']='unitless'
        ds = ds.transpose('time_counter','deptht','yb','xbt')
    if pos == 'U':
        arrayX['gdepu'] = mesh_mask['gdept_0']
        arrayX['e3u'] = mesh_mask['e3u_0']
        if side in ['north', 'south']:
            arrayX = arrayX.isel(X=slice(None,-1))
            nba = 'nbiu'
            nbb = 'nbju'
        if side in ['east', 'west']:
            nba = 'nbju'
            nbb = 'nbiu'
        if side in ['north']:
            arrayX = arrayX.isel(X=slice(None,None,-1))
        if side in ['west']:
            arrayX = arrayX.isel(Y=slice(None,None,-1))
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbu'], arrayX[dim].values + 1),
                         nbb: (['xbu'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbru': (['xbu'], np.full(arrayX[dim].shape, 1 + offset)),
                         'depthu':(['depthu'], arrayX.depthu.values),
                         'nav_lat':(['xbu'], arrayX.nav_lat.values),
                         'nav_lon':(['xbu'], arrayX.nav_lon.values),
              'gdepu':(['depthu','xbu'], arrayX.gdepu.values),
                  'e3u':(['depthu','xbu'], arrayX.e3u.values),
        'vozocrtx':  (['time_counter','depthu','xbu'], arrayX.vozocrtx.values)}
                        ).expand_dims('yb')
        ds.depthu.attrs['long_name']='Vertical U levels'
        ds.depthu.attrs['units']='m'
        ds.vozocrtx.attrs['long_name']='Zonal velocity'
        ds.vozocrtx.attrs['units']='m/s'
        #ds.vozocrtx.attrs['missing_value']= 0
        ds.nbru.attrs['long_name']='bdy discrete distance'
        ds.nbru.attrs['units']='unitless'
        ds.nbju.attrs['long_name']='bdy j index'
        ds.nbju.attrs['units']='unitless'
        ds.nbiu.attrs['long_name']='bdy i index'
        ds.nbiu.attrs['units']='unitless'
        ds['vozocrtx'] = ds.vozocrtx.fillna(0.0)
        ds = ds.transpose('time_counter','depthu','yb','xbu')
    if pos == 'V':
        arrayX['gdepv'] = mesh_mask['gdept_0']
        arrayX['e3v'] = mesh_mask['e3v_0']
        if side in ['west', 'east']:
            arrayX = arrayX.isel(Y=slice(None,-1))
            nba = 'nbjv'
            nbb = 'nbiv'
        if side in ['north', 'south']:
            nba = 'nbiv'
            nbb = 'nbjv'
        if side in ['north']:
            arrayX = arrayX.isel(X=slice(None,None,-1))
        if side in ['west']:
            arrayX = arrayX.isel(Y=slice(None,None,-1))
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbv'], arrayX[dim].values + 1),
                         nbb: (['xbv'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbrv': (['xbv'], np.full(arrayX[dim].shape, 1 + offset)),
                         'depthv':(['depthv'], arrayX.depthv.values),
                         'nav_lon':(['xbv'], arrayX.nav_lon.values),
                         'nav_lat':(['xbv'], arrayX.nav_lat.values),
                'gdepv':(['depthv','xbv'], arrayX.gdepv.values),
                    'e3v':(['depthv','xbv'], arrayX.e3v.values),
        'vomecrty':  (['time_counter','depthv','xbv'], arrayX.vomecrty.values)}
                       ).expand_dims('yb')
        ds.depthv.attrs['long_name']='Vertical V levels'
        ds.depthv.attrs['units']='m'
        ds.vomecrty.attrs['long_name']='Meridional velocity'
        ds.vomecrty.attrs['units']='m/s'
        #ds.vomecrty.attrs['missing_value']= 0
        ds.nbrv.attrs['long_name']='bdy discrete distance'
        ds.nbrv.attrs['units']='unitless'
        ds.nbjv.attrs['long_name']='bdy j index'
        ds.nbjv.attrs['units']='unitless'
        ds.nbiv.attrs['long_name']='bdy i index'
        ds.nbiv.attrs['units']='unitless'
        ds['vomecrty'] = ds.vomecrty.fillna(0.0)
        ds = ds.transpose('time_counter','depthv','yb','xbv')
    ds.nav_lat.attrs['units']='degrees_north'
    ds.nav_lon.attrs['units']='degrees_east'
    ds.attrs['history'] = 'Created using RDPs NEMO config on SCIHUB'
    return ds

def single_bound(data, mesh_mask, side, pos, width=1):
    if width == 1:
        ds = get_side(data, side, pos, offset=0)
    else:
        segments = []
        for i in range(0,width):
            segments.append(get_side(data, side, pos, offset=i))
        ds = xr.concat(segments, dim=('xb' + pos).lower())
    return ds

def get_ring(pos, width):
    segments = []
    for ring in range(0,width):
        print ('RING', ring)
        for side in ['east', 'north', 'west', 'south']:
            if pos == 'T':
                data_pathT = '../Masks/BdyData/bdy_T_' + side + '_masked.nc'
                data = xr.open_dataset(data_pathT, decode_times=False)
            if pos == 'U':
                data_pathU = '../Masks/BdyData/bdy_U_' + side + '_masked.nc'
                data = xr.open_dataset(data_pathU, decode_times=False)
            if pos == 'V':
                data_pathV = '../Masks/BdyData/bdy_V_' + side + '_masked.nc'
                data = xr.open_dataset(data_pathV, decode_times=False)
            #orcaT_path = '../processORCA12/DataIn/ORCA0083-N06_20150105d05T.nc'
            #orca_time = xr.open_dataset(orcaT_path).time_counter
            #print (data)
            #data['time_counter'] = orca_time
            segments.append(get_side(data, side, pos, offset=ring))
    return xr.concat(segments, dim=('xb' + pos).lower())

def full_bounds(width):
    #mesh_mask = xr.open_dataset('mesh_mask.nc')#.isel(
    #              x=slice(None,10), y=slice(1,99)).rename(
    #            {'x':'X','y':'Y'}).rename({'time_counter':'tc'})
    dsT = get_ring('T', width=width)
    dsU = get_ring('U', width=width)
    dsV = get_ring('V', width=width)
    dsT.to_netcdf('BdyOut/bdy_T_ring.nc', unlimited_dims='time_counter')
    dsU.to_netcdf('BdyOut/bdy_U_ring.nc', unlimited_dims='time_counter')
    dsV.to_netcdf('BdyOut/bdy_V_ring.nc', unlimited_dims='time_counter')
full_bounds(20)

def all_pos_one_side(side, width=1):
    data_pathT = '../Masks/BdyData/bdy_T_west_masked.nc'
    dataT = xr.open_dataset(data_pathT, decode_times=False)
    data_pathU = '../Masks/BdyData/bdy_U_west_masked.nc'
    dataU = xr.open_dataset(data_pathU, decode_times=False)
    data_pathV = '../Masks/BdyData/bdy_V_west_masked.nc'
    dataV = xr.open_dataset(data_pathV, decode_times=False)
    orcaT_path = '../processORCA12/DataIn/ORCA0083-N06_20150105d05T.nc'
    orca_time = xr.open_dataset(orcaT_path).time_counter
    mesh_mask = xr.open_dataset('mesh_mask.nc').isel(
                  x=slice(None,10), y=slice(1,99)).rename(
                {'x':'X','y':'Y'}).rename({'time_counter':'tc'})
    dataT['time_counter'] = orca_time
    dataU['time_counter'] = orca_time
    dataV['time_counter'] = orca_time
    dataT['gdept'] = mesh_mask['gdept_0']
    dataU['gdepu'] = mesh_mask['gdept_0']
    dataV['gdepv'] = mesh_mask['gdept_0']
    dataT['e3t'] = mesh_mask['e3t_0']
    dataU['e3u'] = mesh_mask['e3u_0']
    dataV['e3v'] = mesh_mask['e3v_0']
    dsT = single_bound(dataT, side, 'T', width=width)
    dsU = single_bound(dataU, side, 'U', width=width)
    dsV = single_bound(dataV, side, 'V', width=width)
    print (dsT)
    print (dsU)
    print (dsV)
    dsT.to_netcdf('BdyOut/bdy_T_west_masked.nc', unlimited_dims='time_counter')
    dsU.to_netcdf('BdyOut/bdy_U_west_masked.nc', unlimited_dims='time_counter')
    dsV.to_netcdf('BdyOut/bdy_V_west_masked.nc', unlimited_dims='time_counter')

   
#bdy_coords = get_ring(coords, 0)
#if rings > 1:
#    for ring in range(1, rings):
#        print ('ring: ', ring)
#        bdy_coords = xr.concat([bdy_coords, get_ring()], dim='xbT')
#
#bdy_coords.to_netcdf('coordinates_bdy.nc')
