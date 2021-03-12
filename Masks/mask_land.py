import xarray as xr
import numpy as np

def mask_bdy(scalar, year, month):
    for bearing in ['north', 'east', 'south', 'west']:
        ds = xr.load_dataset('BdyData/bdy_'+ scalar +'_' + bearing + 
                             '_y' + year + 'm' + month + '.nc')

        if scalar is 'T':
            ds['vosaline'] = ds.vosaline.fillna(34.0)
            ds['votemper'] = ds.votemper.fillna(0.0)
#        if scalar is 'Tsurf':
            ds['sossheig'] = ds.sossheig.fillna(0.0)
        if scalar is 'U':
            ds['vozoctrx'] = ds.vozocrtx.fillna(0.0)
        if scalar is 'V':
            ds['vomectry'] = ds.vomecrty.fillna(0.0)

        ds.to_netcdf('BdyData/bdy_'+ scalar +'_' + bearing + 
                     '_y' + year + 'm' + month +  '_masked.nc')

mask_bdy('V', '2015', '01')
mask_bdy('T', '2015', '01')
mask_bdy('U', '2015', '01')
#mask_bdy('V', '2014', '12')
#mask_bdy('T', '2014', '12')
#mask_bdy('U', '2014', '12')

#def pad_bounds(scalar):
#    ds[scalar][:,:,-1] = ds[scalar][:,:,-2]
#    ds[scalar][:,:,0]  = ds[scalar][:,:,1]
#    ds[scalar][:,-1] = ds[scalar][:,-2]
#    ds[scalar][:,0]  = ds[scalar][:,1]
#    return ds

#ds = pad_bounds('votemper')
#ds = pad_bounds('vosaline')

