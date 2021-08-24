import xarray as xr
import config

def rename_vars(case, appends):
    data_path = config.data_path() + case + '/'
    
    def rename(ds, stat_type):
        print (ds)
        for key in ds.data_vars:
            print (key)
            ds = ds.rename({key:key + stat_type})
        return ds

    for append in appends:
        ds = xr.open_dataset(data_path + append)
        stat_type = '_' + append.split('_')[2]
        ds = rename(ds, stat_type)
        ds.to_netcdf(data_path + 'Stats/' + append)


appends = ['SOCHIC_PATCH_mean_grid_T.nc',
           'SOCHIC_PATCH_mean_grid_U.nc',
           'SOCHIC_PATCH_mean_grid_V.nc',
           'SOCHIC_PATCH_mean_grid_W.nc',
           'SOCHIC_PATCH_mean_icemod.nc',
           'SOCHIC_PATCH_std_grid_T.nc',
           'SOCHIC_PATCH_std_grid_U.nc',
           'SOCHIC_PATCH_std_grid_V.nc',
           'SOCHIC_PATCH_std_grid_W.nc',
           'SOCHIC_PATCH_std_icemod.nc']

rename_vars('EXP08', appends)
