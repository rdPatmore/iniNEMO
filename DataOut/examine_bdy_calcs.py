import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib

bdy = xr.open_dataset('ORCA24/bdy_V_ring_y2013.nc').isel(time_counter=0,yb=0)
print (bdy)
#bdy = bdy.where(bdy.nbrv == 15)

plt.figure()
length = len(bdy.nav_lat)
print (bdy.nav_lat)
print (np.arange(length))
plt.scatter(bdy.nav_lat, bdy.nav_lon, c=np.arange(length))#,
            #cmap=plt.cm.plasma(np.linspace(0,1,length)))
plt.savefig('bdy_test.png')
