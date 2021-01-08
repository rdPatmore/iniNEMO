import xarray as xr
import numpy  as np

coords = xr.open_dataset('coordinates.nc', decode_times=False)

def get_side(side, pos, offset=0):
    if side == 'west':
        arrayX = coords.isel(X=0 + offset, Y=slice(1,99))  
    if side == 'east':
        arrayX = coords.isel(X=-1 - offset)  
    if side == 'south':
        arrayX = coords.isel(Y=0 + offset)  
    if side == 'north':
        arrayX = coords.isel(Y=-1 - offset)  

    dim='Y'
    if pos == 'T':
        ds = xr.Dataset({'nbjt': (['xbt'], arrayX[dim].values),
                         'nbit': (['xbt'], np.full(arrayX[dim].shape,
                                                   arrayX.X.values+1)),
                         'nbrt': (['xbt'], np.full(arrayX[dim].shape, 1)),
                         'glamt':(['xbt'], arrayX.glamt.values),
                         'gphit':(['xbt'], arrayX.gphit.values),
                         'e1t':  (['xbt'], arrayX.e1t.values),
                         'e2t':  (['xbt'], arrayX.e2t.values)}
                       ).expand_dims('yb')
    if pos == 'U':
        ds = xr.Dataset({'nbju': (['xbu'], arrayX[dim].values),
                         'nbiu': (['xbu'], np.full(arrayX[dim].shape,
                                                   arrayX.X.values +1)),
                         'nbru': (['xbu'], np.full(arrayX[dim].shape, 1)),
                         'glamu':(['xbu'], arrayX.glamu.values),
                         'gphiu':(['xbu'], arrayX.gphiu.values),
                         'e1u':  (['xbu'], arrayX.e1u.values),
                         'e2u':  (['xbu'], arrayX.e2u.values)}
                        ).expand_dims('yb')
    if pos == 'V':
        arrayX = arrayX.isel(Y=slice(None,-1))
        print (arrayX)
        ds = xr.Dataset({'nbjv': (['xbv'], arrayX[dim].values),
                         'nbiv': (['xbv'], np.full(arrayX[dim].shape,
                                                   arrayX.X.values+1)),
                         'nbrv': (['xbv'], np.full(arrayX[dim].shape, 1)),
                         'glamv':(['xbv'], arrayX.glamv.values),
                         'gphiv':(['xbv'], arrayX.gphiv.values),
                         'e1v':  (['xbv'], arrayX.e1v.values),
                         'e2v':  (['xbv'], arrayX.e2v.values)}
                       ).expand_dims('yb')
    return ds
#def get_side(side, offset=0):
#    if side == 'west':
#        arrayT = coords.isel(X=0 + offset)  
#        arrayU = coords.isel(X=0 + offset)  
#        arrayV = coords.isel(X=0 + offset)  
#    if side == 'east':
#        arrayT = coords.isel(X=-1 - offset)  
#        arrayU = coords.isel(X=-1 - offset)  
#        arrayV = coords.isel(X=-1 - offset)  
#    if side == 'south':
#        arrayT = coords.isel(Y=0 + offset)  
#        arrayU = coords.isel(Y=0 + offset)  
#        arrayV = coords.isel(Y=0 + offset)  
#    if side == 'north':
#        arrayT = coords.isel(Y=-1 - offset)  
#        arrayU = coords.isel(Y=-1 - offset)  
#        arrayV = coords.isel(Y=-1 - offset)  
#
#    dim='Y'
#    arrayT = xr.Dataset({'nbit': (['xbt'], arrayT[dim].values),
#                         'nbjt': (['xbt'], np.full(arrayT[dim].shape,
#                                                   arrayT.X.values)),
#                         'nbrt': (['xbt'], np.full(arrayT[dim].shape, 1)),
#                         'glamt':(['xbt'], arrayT.glamt.values),
#                         'gphit':(['xbt'], arrayT.gphit.values),
#                         'e1t':  (['xbt'], arrayT.e1t.values),
#                         'e2t':  (['xbt'], arrayT.e2t.values)}
#                       ).expand_dims('yb')
#    arrayU = xr.Dataset({'nbiu': (['xbu'], arrayU[dim].values),
#                         'nbju': (['xbu'], np.full(arrayU[dim].shape,
#                                                   arrayU.X.values)),
#                         'nbru': (['xbu'], np.full(arrayU[dim].shape, 1)),
#                         'glamu':(['xbu'], arrayU.glamu.values),
#                         'gphiu':(['xbu'], arrayU.gphiu.values),
#                         'e1u':  (['xbu'], arrayU.e1u.values),
#                         'e2u':  (['xbu'], arrayU.e2u.values)}
#                       ).expand_dims('yb')
#    arrayV = xr.Dataset({'nbiv': (['xbv'], arrayV[dim].values),
#                         'nbjv': (['xbv'], np.full(arrayV[dim].shape,
#                                                   arrayV.X.values)),
#                         'nbrv': (['xbv'], np.full(arrayV[dim].shape, 1)),
#                         'glamv':(['xbv'], arrayV.glamv.values),
#                         'gphiv':(['xbv'], arrayV.gphiv.values),
#                         'e1v':  (['xbv'], arrayV.e1v.values),
#                         'e2v':  (['xbv'], arrayV.e2v.values)}
#                       ).expand_dims('yb')
#    ds = xr.merge([arrayT, arrayU, arrayV])
#    return ds

def single_bound(side, pos, width=1):
    if width == 1:
        ds = get_side(side, pos, offset=0)
    else:
        segments = []
        for i in range(0,width):
            segments.append(get_side(side, pos, offset=i))
        ds = xr.concat(segments, dim=('xb' + pos).lower())
    return ds

def merge_pos(side, width=1):
    dsT = single_bound(side, 'T', width=width)
    dsU = single_bound(side, 'U', width=width)
    dsV = single_bound(side, 'V', width=width)
    ds = xr.merge([dsT, dsU, dsV])
    print (ds)
    ds.to_netcdf('coordinates.bdy.nc')
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
