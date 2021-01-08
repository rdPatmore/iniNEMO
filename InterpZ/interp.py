import xarray as xr
import numpy as np

# read domain_cfg and ini_state
ds_cfg   = xr.open_dataset('domain_cfg.nc')
ds_state = xr.open_dataset('ini_state.nc')
print (ds_cfg.nav_lev)
print (ds_state.depth)

# assign depth coords
ds_state = ds_state.assign_coords(depth=ds_state.depth)
ds_state = ds_state.swap_dims({'Z':'depth'})

# interpolate
interpolated = ds_state.interp(depth=ds_cfg.nav_lev)
interpolated = interpolated.reset_coords('depth')
interpolated = interpolated.assign_coords(
               {'Z': ('z', 
                      np.arange(1, ds_cfg.nav_lev.shape[0] + 1, dtype=np.int32),
                      ds_state.Z.attrs)})
interpolated = interpolated.swap_dims({'z':'Z'})

masked_interpolated = interpolated.fillna(0.0)
masked_interpolated['votemper'][-1] = 0.0
masked_interpolated['vosaline'][-1] = 0.0

# save
masked_interpolated.to_netcdf('ini_state_masked_interpolated.nc')

