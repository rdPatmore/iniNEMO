import xarray as xr
import numpy as np
import datetime

T = xr.open_dataset('BdyOut/bdy_T_ring.nc')
U = xr.open_dataset('BdyOut/bdy_U_ring.nc')
V = xr.open_dataset('BdyOut/bdy_V_ring.nc')
new_time = np.array(['2015-01-01T00:00:00.000000000', '2015-01-06T00:00:00.000000000',
            '2015-01-11T00:00:00.000000000', '2015-01-16T00:00:00.000000000',
            '2015-01-21T00:00:00.000000000', '2015-01-26T00:00:00.000000000'],
            dtype='datetime64')

T = T.assign_coords(time_counter=new_time)
U = U.assign_coords(time_counter=new_time)
V = V.assign_coords(time_counter=new_time)
T = T.drop('T')
U = U.drop('T')
V = V.drop('T')
T = T.rename_dims({'time_counter':'time'})
U = U.rename_dims({'time_counter':'time'})
V = V.rename_dims({'time_counter':'time'})
T = T.rename({'time_counter':'time'})
U = U.rename({'time_counter':'time'})
V = V.rename({'time_counter':'time'})


T.time.encoding['dtype'] = np.float64
U.time.encoding['dtype'] = np.float64
V.time.encoding['dtype'] = np.float64
T.to_netcdf('BdyOut/bdy_T_ring_early.nc', unlimited_dims='time')
U.to_netcdf('BdyOut/bdy_U_ring_early.nc', unlimited_dims='time')
V.to_netcdf('BdyOut/bdy_V_ring_early.nc', unlimited_dims='time')
