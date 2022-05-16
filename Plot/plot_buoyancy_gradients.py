import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
from get_transects import get_transects

dask.config.set({"array.slicing.split_large_chunks": True})

matplotlib.rcParams.update({'font.size': 8})

class plot_buoyancy_gradients(object):
    '''
    for ploting results of adjusting the geometry of the glider path
    '''

    def __init__(self, case, offset=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

    def model_buoyancy_gradients_at_depth(self, depth=10):
    
        # model
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc',
                             chunks='auto')
        bg = np.abs(bg.sel(deptht=depth, method='nearest'))

        bg = bg.sel(time_counter='2013-01-01', method='nearest')

        fig, ax= plt.subplots(1, figsize=(4.5,4.5))
        ax.pcolor(bg, cmap=plt.cm.RdBu_r, vmin=-1e-8, vmax=1e-8) 
        plt.show()

p = plot_buoyancy_gradients('EXP10')
p.model_buoyancy_gradients_at_depth()

