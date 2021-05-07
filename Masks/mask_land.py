import xarray as xr
import numpy as np

def mask_bdy(scalar, year, month, day):
    indir = '/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_12'
    for bearing in ['north', 'east', 'south', 'west']:
        ds = xr.load_dataset(indir + '/bdy_'+ scalar +'_' + bearing + 
                             '_y' + year + 'm' + month + 'd' + day + '.nc')

        print (scalar)
        if scalar == 'T':
            ds['vosaline'] = ds.vosaline.fillna(34.0)
            ds['votemper'] = ds.votemper.fillna(0.0)
#        if scalar is 'Tsurf':
            ds['sossheig'] = ds.sossheig.fillna(0.0)
        if scalar == 'U':
            ds['vozoctrx'] = ds.vozocrtx.fillna(0.0)
        if scalar == 'V':
            ds['vomectry'] = ds.vomecrty.fillna(0.0)
        if scalar == 'I':
            ds['siconc'] = ds.siconc.fillna(0.0)
            ds['sithic'] = ds.sithic.fillna(0.0)
            ds['snthic'] = ds.snthic.fillna(0.0)

        ds.to_netcdf('../DataOut/bdy_'+ scalar +'_' + bearing + 
                     '_y' + year + '_masked.nc')

mask_bdy('V', '2012', '01', '03')
mask_bdy('T', '2012', '01', '03')
mask_bdy('U', '2012', '01', '03')
mask_bdy('I', '2012', '01', '03')
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

