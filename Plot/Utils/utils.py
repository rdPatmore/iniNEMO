import numpy as np
import xarray as xr

class funcs(object):

    def __init__(self):

        self.hist_range=(0,2e-8)

    def get_bg_z_hist(self, bg, bins=20):
        ''' calculate histogram and assign to xarray dataset '''
    
        # stack dimensions
        stacked_bgx = bg.bx.stack(z=('time_counter','x','y'))
        stacked_bgy = bg.by.stack(z=('time_counter','x','y'))
    
        # bg norm - warning: not gridded appropriately on T-pts
        stacked_bg_norm = (stacked_bgx**2 + stacked_bgy**2)**0.5
    
        # histogram
        hist_x, bins = np.histogram(stacked_bgx.dropna('z', how='all'),
                            range=self.hist_range, density=True, bins=bins)
        hist_y, bins = np.histogram(stacked_bgy.dropna('z', how='all'),
                            range=self.hist_range, density=True, bins=bins)
        hist_norm, bins = np.histogram(
                            stacked_bg_norm.dropna('z', how='all'),
                            range=self.hist_range, density=True, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        # assign to dataset
        hist_ds = xr.Dataset({'hist_x':(['bin_centers'], hist_x),
                              'hist_y':(['bin_centers'], hist_y),
                              'hist_norm':(['bin_centers'], hist_norm)},
                   coords={'bin_centers': (['bin_centers'], bin_centers),
                           'bin_left'   : (['bin_centers'], bins[:-1]),
                           'bin_right'  : (['bin_centers'], bins[1:])})
        return hist_ds
