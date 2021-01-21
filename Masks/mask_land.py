import xarray as xr
import numpy as np

def mask_bdy(scalar):
    for bearing in ['north', 'east', 'south', 'west']:
        ds = xr.load_dataset('BdyData/bdy_'+ scalar +'_' + bearing + '.nc')

        if scalar is 'T':
            ds['vosaline'] = ds.vosaline.fillna(0.0)
            ds['votemper'] = ds.votemper.fillna(0.0)
#        if scalar is 'Tsurf':
            ds['sossheig'] = ds.sossheig.fillna(0.0)
        if scalar is 'U':
            ds['vozoctrx'] = ds.vozocrtx.fillna(0.0)
        if scalar is 'V':
            ds['vomectry'] = ds.vomecrty.fillna(0.0)

        ds.to_netcdf('BdyData/bdy_'+ scalar +'_' + bearing + '_masked.nc')

mask_bdy('V')
mask_bdy('T')
mask_bdy('U')

#def pad_bounds(scalar):
#    ds[scalar][:,:,-1] = ds[scalar][:,:,-2]
#    ds[scalar][:,:,0]  = ds[scalar][:,:,1]
#    ds[scalar][:,-1] = ds[scalar][:,-2]
#    ds[scalar][:,0]  = ds[scalar][:,1]
#    return ds

#ds = pad_bounds('votemper')
#ds = pad_bounds('vosaline')

