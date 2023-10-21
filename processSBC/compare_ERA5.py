import xarray as xr

""" compare nico's netcdf output with mine """

nico  = '/projectsa/NEMO/ryapat/JASMIN/'
rdp   = '/projectsa/NEMO/ryapat/Forcing/'
fname = 'ERA5_U10_y'

dsn = xr.open_dataset(nico + fname + '2004.nc')
dsr = xr.open_dataset(rdp  + fname + '1999.nc')

print (dsn)
print ('')
print ('')
print ('')
print (dsr)
