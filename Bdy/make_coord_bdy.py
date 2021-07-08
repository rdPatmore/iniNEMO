import xarray as xr
import numpy  as np

path = '/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_48/'
coords = xr.open_dataset(path + 'coordinates.nc',
                         decode_times=False)#.rename({
       # 'x':'X', 'y':'Y'}).squeeze('time_counter').reset_coords('time_counter')
coords = coords.isel(X=slice(1,-1),Y=slice(1,-1))
print (coords.sizes)

def get_side(side, pos, offset=0):
    '''
    process a SIREN bdy file
    '''

    xlen = coords.sizes['X']
    ylen = coords.sizes['Y'] 
    print (xlen)
    print (ylen)

    print ('SIDE: ', side)
    vel_shift=0
    #if pos == 'I':
    #    vel_shift=1
    if side == 'west':
        print (coords.isel(X=offset).sizes)
        arrayX = coords.isel(X=offset, Y=slice(0, ylen))
        #arrayX = coords.isel(X=offset, Y=slice(0+offset, ylen-offset))
        dim='Y'
        bdy_pos = 2 + offset
        print (arrayX.sizes)

    if side == 'east':
        if pos == 'U':
            vel_shift=1
        arrayX = coords.isel(X=-1-offset-vel_shift, Y=slice(0, ylen))
        #arrayX = coords.isel(X=-1-offset-vel_shift,
        #                     Y=slice(0+offset, ylen-offset))
        bdy_pos = xlen +1- offset - vel_shift
        print (arrayX.sizes)
        dim='Y'

    if side == 'south':
        #arrayX = coords.isel(Y=offset, X=slice(0+offset, xlen-offset))
        arrayX = coords.isel(Y=offset, X=slice(0, xlen))
        dim='X'
        bdy_pos = 2 + offset

    if side == 'north':
        if pos == 'V':
            vel_shift=1
        arrayX = coords.isel(Y=-1-offset-vel_shift, X=slice(0, xlen))
        #arrayX = coords.isel(Y=-1-offset-vel_shift, 
        #                     X=slice(0+offset, xlen-offset))
        bdy_pos = ylen + 1 - offset - vel_shift
        dim='X'
        #print (' ')
        #print (' ')
        #print (' ')
        #print (' ')
        #print (' ****** offset *****: ', offset)
        #print (arrayX.nav_lat)
        #print (' ')
        #print (' ')
        #print (' ****** offset *****: ', offset)
        #print (arrayX.nav_lon)
        #print (' ')
        #print (' ')
        ##print (' ')
        ##print (' ')
        ##print (' ')

    if pos == 'T':
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
        print (arrayX[dim].values)
        print (np.full(arrayX[dim].shape, bdy_pos))
        ds = xr.Dataset({nba: (['xbt'], arrayX[dim].values),
                         nbb: (['xbt'], np.full(arrayX[dim].shape, bdy_pos)),
                       'nbrt': (['xbt'], np.full(arrayX[dim].shape,offset+1)),
                         'glamt':(['xbt'], arrayX.glamt.values),
                         'gphit':(['xbt'], arrayX.gphit.values),
                         'e1t':  (['xbt'], arrayX.e1t.values),
                         'e2t':  (['xbt'], arrayX.e2t.values)}
                       ).expand_dims('yb')
        ds.nbrt.attrs['long_name']='bdy discrete distance'
        ds.nbrt.attrs['units']='unitless'
        ds.nbjt.attrs['long_name']='bdy j index'
        ds.nbjt.attrs['units']='unitless'
        ds.nbit.attrs['long_name']='bdy i index'
        ds.nbit.attrs['units']='unitless'
        ds = ds.transpose('yb','xbt')

    if pos == 'U':
        if side in ['north', 'south']:
            nba = 'nbiu'
            nbb = 'nbju'
        if side in ['east', 'west']:
            nba = 'nbju'
            nbb = 'nbiu'
        if side in ['north']:
            arrayX = arrayX.sortby('X', ascending=False)
        if side in ['west']:
            arrayX = arrayX.sortby('Y', ascending=False)
        ds = xr.Dataset({nba: (['xbu'], arrayX[dim].values),
                         nbb: (['xbu'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbru': (['xbu'], np.full(arrayX[dim].shape,offset+1)),
                         'glamu':(['xbu'], arrayX.glamu.values),
                         'gphiu':(['xbu'], arrayX.gphiu.values),
                         'e1u':  (['xbu'], arrayX.e1u.values),
                         'e2u':  (['xbu'], arrayX.e2u.values)}
                        ).expand_dims('yb')
        ds.nbru.attrs['long_name']='bdy discrete distance'
        ds.nbru.attrs['units']='unitless'
        ds.nbju.attrs['long_name']='bdy j index'
        ds.nbju.attrs['units']='unitless'
        ds.nbiu.attrs['long_name']='bdy i index'
        ds.nbiu.attrs['units']='unitless'
        ds = ds.transpose('yb','xbu')

    if pos == 'V':
        if side in ['west', 'east']:
            #arrayX = arrayX.isel(Y=slice(None,-1))
            nba = 'nbjv'
            nbb = 'nbiv'
        if side in ['north', 'south']:
            nba = 'nbiv'
            nbb = 'nbjv'
        if side in ['north']:
            arrayX = arrayX.sortby('X', ascending=False)
        if side in ['west']:
            arrayX = arrayX.sortby('Y', ascending=False)
        ds = xr.Dataset({nba: (['xbv'], arrayX[dim].values),
                         nbb: (['xbv'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbrv': (['xbv'], np.full(arrayX[dim].shape,offset+1)),
                         'glamv':(['xbv'], arrayX.glamv.values),
                         'gphiv':(['xbv'], arrayX.gphiv.values),
                         'e1v':  (['xbv'], arrayX.e1v.values),
                         'e2v':  (['xbv'], arrayX.e2v.values)}
                       ).expand_dims('yb')
        ds.nbrv.attrs['long_name']='bdy discrete distance'
        ds.nbrv.attrs['units']='unitless'
        ds.nbjv.attrs['long_name']='bdy j index'
        ds.nbjv.attrs['units']='unitless'
        ds.nbiv.attrs['long_name']='bdy i index'
        ds.nbiv.attrs['units']='unitless'
        ds = ds.transpose('yb','xbv')
    ds.attrs['history'] = 'Created using RDPs NEMO config on SCIHUB'
    return ds

def single_bound(side, pos, width=1):
    if width == 1:
        ds = get_side(side, pos, offset=0)
    else:
        segments = []
        for i in range(0,width):
            segments.append(get_side(side, pos, offset=i))
        ds = xr.concat(segments, dim=('xb' + pos).lower())
    return ds

def get_ring(pos, width):
    '''
    two loops:
        1. join sides to create a full boundary={east, north, west, south}
        2. loop over width of boundary conditions
    '''

    segments = []
    for ring in range(0,width):
        print ('RING', ring)
        for side in ['east', 'north', 'west', 'south']:
            segments.append(get_side(side, pos, offset=ring))
    return xr.concat(segments, dim=('xb' + pos).lower())


def full_bounds(width=0):
    '''
    write coordinates.bdy.nc
    full boundary condition coordinates of width=width for {T,U,V}
    '''

    dsT = get_ring('T', width=width)
    dsU = get_ring('U', width=width)
    dsV = get_ring('V', width=width)
    ds = xr.merge([dsT, dsU, dsV])
    ds.to_netcdf('../DataOut/ORCA48/coordinates.bdy.nc')

full_bounds(width=20)

def merge_pos(side, width=1):
    dsT = single_bound(side, 'T', width=width)
    dsU = single_bound(side, 'U', width=width)
    dsV = single_bound(side, 'V', width=width)
    ds = xr.merge([dsT, dsU, dsV])
    print (ds)
    ds.to_netcdf('coordinates.bdy.nc')

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
