import xarray as xr
import numpy  as np


def get_side(data, side, pos, offset=0):
    if side == 'west':
        arrayX = data.isel(X=0 + offset).reset_coords('X', drop=True)
    if side == 'east':
        arrayX = data.isel(X=-1 - offset)  
    if side == 'south':
        arrayX = data.isel(Y=0 + offset)  
    if side == 'north':
        arrayX = data.isel(Y=-1 - offset)  
    #arrayX = arrayX.squeeze({'T'})
    dim='Y'
    if pos == 'T':
        #print (arrayX.gdept)
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         'nbjt': (['xbt'], arrayX[dim].values + 1),
                         'nbit': (['xbt'], np.full(arrayX[dim].shape,
                                                 int(arrayX.attrs['bdy_deb']))),
                         'nbrt': (['xbt'], np.full(arrayX[dim].shape, 1)),
                         'deptht':(['deptht'], arrayX.deptht.values),
                         'nav_lon':(['xbt'], arrayX.nav_lon.values),
                         'nav_lat':(['xbt'], arrayX.nav_lat.values),
                 'gdept':(['time_counter','deptht','xbt'], arrayX.gdept.values),
                     'e3t':(['time_counter','deptht','xbt'], arrayX.e3t.values),
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
        ds = xr.Dataset({'nbju': (['xbu'], arrayX[dim].values + 1),
                         'nbiu': (['xbu'], np.full(arrayX[dim].shape,
                                                 int(arrayX.attrs['bdy_deb']))),
                         'nbru': (['xbu'], np.full(arrayX[dim].shape, 1)),
                         'depthu':(['depthu'], arrayX.depthu.values),
                         'nav_lat':(['xbu'], arrayX.nav_lat.values),
                         'nav_lon':(['xbu'], arrayX.nav_lon.values),
              'gdepu':(['time_counter','depthu','xbu'], arrayX.gdepu.values),
                  'e3u':(['time_counter','depthu','xbu'], arrayX.e3u.values),
        'vozocrtx':  (['time_counter','depthu','xbu'], arrayX.vozocrtx.values)}
                        ).expand_dims('yb')
        ds.depthu.attrs['long_name']='Vertical U levels'
        ds.depthu.attrs['units']='m'
        ds.vozocrtx.attrs['long_name']='Zonal velocity'
        ds.vozocrtx.attrs['units']='m/s'
        ds.vozocrtx.attrs['missing_value']= 0
        ds.nbru.attrs['long_name']='bdy discrete distance'
        ds.nbru.attrs['units']='unitless'
        ds.nbju.attrs['long_name']='bdy j index'
        ds.nbju.attrs['units']='unitless'
        ds.nbiu.attrs['long_name']='bdy i index'
        ds.nbiu.attrs['units']='unitless'
        ds['vozocrtx'] = ds.vozocrtx.fillna(0.0)
        ds = ds.transpose('time_counter','depthu','yb','xbu')
    if pos == 'V':
        arrayX = arrayX.isel(Y=slice(None,-1))
        ds = xr.Dataset({'nbjv': (['xbv'], arrayX[dim].values + 1),
                         'nbiv': (['xbv'], np.full(arrayX[dim].shape,
                                              int(arrayX.attrs['bdy_deb']))),
                         'nbrv': (['xbv'], np.full(arrayX[dim].shape, 1)),
                         'depthv':(['depthv'], arrayX.depthv.values),
                         'nav_lon':(['xbv'], arrayX.nav_lon.values),
                         'nav_lat':(['xbv'], arrayX.nav_lat.values),
                'gdepv':(['time_counter','depthv','xbv'], arrayX.gdepv.values),
                    'e3v':(['time_counter','depthv','xbv'], arrayX.e3v.values),
        'vomecrty':  (['time_counter','depthv','xbv'], arrayX.vomecrty.values)}
                       ).expand_dims('yb')
        ds.depthv.attrs['long_name']='Vertical V levels'
        ds.depthv.attrs['units']='m'
        ds.vomecrty.attrs['long_name']='Meridional velocity'
        ds.vomecrty.attrs['units']='m/s'
        ds.vomecrty.attrs['missing_value']= 0
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

def single_bound(data, side, pos, width=1):
    if width == 1:
        ds = get_side(data, side, pos, offset=0)
    else:
        segments = []
        for i in range(0,width):
            segments.append(get_side(data, side, pos, offset=i))
        ds = xr.concat(segments, dim=('xb' + pos).lower())
    return ds

def merge_pos(side, width=1):
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
    #print (dsT)
    #print (dsU)
    #print (dsV)
    dsT.to_netcdf('BdyOut/bdy_T_west_masked.nc', unlimited_dims='time_counter')
    dsU.to_netcdf('BdyOut/bdy_U_west_masked.nc', unlimited_dims='time_counter')
    dsV.to_netcdf('BdyOut/bdy_V_west_masked.nc', unlimited_dims='time_counter')
merge_pos('west', width=1)

def get_ring(coords, pos):
    coords = coords.isel(T=0)
    w = coords.isel(X=0 + pos).reset_coords(['T','X','Y'])
    e = coords.isel(X=-1 - pos).reset_coords(['T','X','Y'])
    s = coords.isel(Y=0 + pos).reset_coords(['T','X','Y'])
    n = coords.isel(Y=-1 - pos).reset_coords(['T','X','Y'])
    print (w)
    print (e)
    print (s)
    print (n)
    return xr.concat([e, n[::-1], w[:,::-1], s], dim='xbT')
   
#bdy_coords = get_ring(coords, 0)
#if rings > 1:
#    for ring in range(1, rings):
#        print ('ring: ', ring)
#        bdy_coords = xr.concat([bdy_coords, get_ring()], dim='xbT')
#
#bdy_coords.to_netcdf('coordinates_bdy.nc')
