import xarray as xr
import numpy as np

def mask_bdy(scalar, year, month, day, res='12'):
    indir = '/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_' + res
    for bearing in ['north', 'east', 'south', 'west']:
        ds = xr.load_dataset(indir + '/bdy_'+ scalar +'_' + bearing + 
                             '_y' + year + 'm' + month + 'd' + day + '.nc')

        print (scalar)
        if scalar == 'T':
            ds['vosaline'] = ds.vosaline.fillna(34.0)
            ds['votemper'] = ds.votemper.fillna(0.0)
            ds['vosaline'] = xr.where(ds.vosaline == 1e20, 34.0, ds.vosaline)
            ds['votemper'] = xr.where(ds.votemper == 1e20, 0.0, ds.votemper)
#        if scalar is 'Tsurf':
            ds['sossheig'] = ds.sossheig.fillna(0.0)
            ds['sossheig'] = xr.where(ds.sossheig == 1e20, 0.0, ds.sossheig)
        if scalar == 'U':
            ds['vozocrtx'] = ds.vozocrtx.fillna(0.0)
            ds['vozocrtx'] = xr.where(ds.vozocrtx == 1e20, 0.0, ds.vozocrtx)
        if scalar == 'V':
            ds['vomecrty'] = ds.vomecrty.fillna(0.0)
            ds['vomecrty'] = xr.where(ds.vomecrty == 1e20, 0.0, ds.vomecrty)
        if scalar == 'I':
            ds['siconc'] = ds.siconc.fillna(0.0)
            ds['sithic'] = ds.sithic.fillna(0.0)
            ds['snthic'] = ds.snthic.fillna(0.0)
            ds['sitemp'] = ds.sitemp.fillna(-3.15)

            ds['siconc'] = xr.where(ds.siconc == 1e20, 0.0, ds.siconc)
            ds['sithic'] = xr.where(ds.sithic == 1e20, 0.0, ds.sithic)
            ds['snthic'] = xr.where(ds.snthic == 1e20, 0.0, ds.snthic)
            ds['sitemp'] = xr.where(ds.sitemp == 1e20, -3.15, ds.sitemp)

        print (ds)
        ds.to_netcdf('../DataOut/ORCA' + res + '/bdy_'+ scalar +'_' + bearing + 
                     '_y' + year + '_masked.nc')

mask_bdy('V', '2013', '01', '03', res='48')
mask_bdy('T', '2013', '01', '03', res='48')
mask_bdy('U', '2013', '01', '03', res='48')
mask_bdy('I', '2013', '01', '03', res='48')
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

