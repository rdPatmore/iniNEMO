import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np

def render_prof(case, ax):
    rho = xr.open_dataset(config.data_path() + '/' + case +
                            '/SOCHIC_PATCH_3h_20121209_20130331_grid_W.nc').bn2
    rho = rho.sel(depthw=slice(0,100))
    rho = rho.isel(x=20,y=20)
    #rho = rho.(['x','y'])
    times = [i for i in range(0,900,10)]
    print (times)
    c = plt.cm.inferno(np.linspace(0,1,len(times)))
    for i, t in enumerate(times):
        rhot = rho.isel(time_counter=t)
        ax.plot(rhot, -rhot.depthw, c=c[i])
    

fig, axs = plt.subplots(3)
render_prof('EXP13', axs[0])
axs[0].set_xlim(-1e-6,1e-6)
plt.show()
