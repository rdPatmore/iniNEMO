import xarray as xr
import numpy as np

def get_montly_means(path, pos, model='', case_num=''):
    # load data
    if model == 'orca':
        data   = xr.open_dataset(orca_path, decode_cf=False,
                                 chunks={'time_counter': 1})
        data = xr.decode_cf(data)
    if model == 'sochic':
        try:
            data = xr.open_mfdataset(sochic_path, decode_cf=True,
                                     chunks={'time_counter': 1})
        except:
            data = xr.open_mfdataset(sochic_path, decode_cf=False, 
                                     chunks={'time_counter': 1})
            #data = data.drop_vars('time_instant')
            data = xr.decode_cf(data)

    # align time steps
    try:
        data['time_counter'] = data.indexes['time_counter'].to_datetimeindex()
    except:
        print ('leap skipping to_datetimeindex')

    print ('loading means')
    month = data.sel(time_counter='2012-01')
    print ('loaded')
    print (data)

    data.to_netcdf('tmp/' + model + '_' + case_num + '_' + pos + '_201201.nc')

case_num = 'EXP08'
outdir = '/work/n02/n02/ryapat30/nemo/nemoHEAD/cfgs/SOCHIC_ICE/'
outpath = outdir + case_num 

#for pos in ['U','V']:
#    print (pos)
##    #sochic_path = outpath + '/SOCHIC_PATCH_3h_' + dates + '_grid_' + pos + '.nc'
#    sochic_path = outpath + '/SOCHIC_PATCH_6h_*_grid_' + pos + '.nc'
#    get_montly_means(sochic_path, pos=pos, model='sochic', case_num=case_num)
##sochic_path = outpath + '/SOCHIC_PATCH_3h_' + dates + '_icemod.nc'
sochic_path = outpath + '/SOCHIC_PATCH_6h_*_icemod.nc'
get_montly_means(sochic_path, pos='I', model='sochic', case_num=case_num)

# T
#orca_path = '../OrcaCutData/ORCA_PATCH_2012_T.nc'
#get_montly_means(orca_path, pos='T', model='orca')

# U
#orca_path = '../OrcaCutData/ORCA_PATCH_2012_U.nc'
#get_montly_means(orca_path, pos='U', model='orca')

# V
#orca_path = '../OrcaCutData/ORCA_PATCH_2012_V.nc'
##get_montly_means(orca_path, pos='V', model='orca')

# I
#orca_path = '../OrcaCutData/ORCA_PATCH_2012_I.nc'
#get_montly_means(orca_path, pos='I', model='orca')
