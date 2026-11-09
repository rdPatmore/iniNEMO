import xarray as xr
import matplotlib.pyplot as plt

bdy = xr.open_dataset('../DataOut/SATLAN/ORCA12/ModelReady/bdy_T_ring_y2012.nc')
coord = xr.open_dataset('/work/n02/n02/ryapat30/nemo/nemoHEAD/cfgs/SOCHIC_ICE/EXP15/domain_cfg.nc').squeeze()

lat = coord.nav_lat
lon = coord.nav_lon
print (coord.bathy_meter)
plt.pcolor(lat,lon,coord.bathy_meter)
plt.scatter(bdy.nav_lat,bdy.nav_lon,s=bdy.nbrt*2, c='r')
plt.savefig('bdy.png')


