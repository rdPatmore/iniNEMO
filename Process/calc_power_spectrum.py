import xarray as xr
import config
import gsw
import scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import window

# not sure on the application here
# however the expectation is that, on average, slopes will reduce with increases
# in resolution as more submesoscale process are resolved.


class power_spectrum(object):

    def __init__(self, model, var):
        self.var = var 
        self.path = config.data_path()
        if var in ['votemper', 'vosaline']:
            self.ds = xr.open_dataset(self.path +  model + 
                            '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc')
            self.ds = self.ds[var]
        self.cfg = xr.open_dataset(self.path + model + '/domain_cfg.nc')

        # remove halo
        self.cfg = self.cfg.isel(x=slice(1,-1), y=slice(1,-1))
        self.ds = self.ds.isel(x=slice(1,-1), y=slice(1,-1))

        # add index
        self.ds = self.ds.assign_coords(i=('x', np.arange(1,self.ds.x.size+1)),
                                        j=('y', np.arange(1,self.ds.y.size+1)))

    def interp_to_regular_grid(self):
        ''' 
        fft requires a regular, square grid to function
        interpolate lat lon to regular cartesian grid
        '''

        x_grid = self.cfg.e1t.isel(y=0).cumsum()
        y_grid = self.cfg.e2t.isel(x=0).cumsum()
      
        # set cartesian grid as dims ready for interp
        self.ds['x'] = x_grid.squeeze()
        self.ds['y'] = y_grid.squeeze()
        #self.ds = self.ds.set_coords(['x', 'y'])
        new_x = xr.DataArray(np.arange(self.ds.x.min(), 
                                       self.ds.x.max(),
                                       self.cfg.e1t.mean()), dims='x')
        new_y = xr.DataArray(np.arange(self.ds.y.min(), 
                                       self.ds.y.max(),
                                       self.cfg.e2t.mean()), dims='y')
        # interp
        self.ds = self.ds.interp(x=new_x, y=new_y)

    def detrend(self):
        ''' remove low wavenumber signals since these are not resolved '''

        Nxm1 = self.ds.x.size - 1
        Nym1 = self.ds.y.size - 1
        x_slope = (self.ds.isel(x=-1) - self.ds.isel(x=0)) / Nxm1       
        y_slope = (self.ds.isel(y=-1) - self.ds.isel(y=0)) / Nym1
     
        Nxp1 = self.ds.x.size + 1
        Nyp1 = self.ds.y.size + 1

        x_detrend  = self.ds   - 0.5 * ( 2*self.ds.i - Nxp1) * x_slope
        xy_detrend = x_detrend - 0.5 * ( 2*self.ds.j - Nyp1) * y_slope

        return xy_detrend
        
    def azimuthalAverage_1(self, image, center=None):
        """
        Calculate the azimuthally averaged radial profile.
    
        image - The 2D image
        center - The [x,y] pixel coordinates used as the center. The default is 
                 None, which then uses the center of the image (including 
                 fracitonal pixels).
        
        """
        # Calculate the indices from the image
        y, x = np.indices(image.shape)
    
        if not center:
            center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
    
        r = np.hypot(x - center[0], y - center[1])
    
        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]
    
        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)
    
        # Find all pixels that fall within each radial bin.
        deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
        rind = np.where(deltar)[0]       # location of changed radius
        nr = rind[1:] - rind[:-1]        # number of radius bin
        
        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
    
        radial_prof = tbin / nr
    
        return radial_prof, len(nr)

    def azimuthalAverage_2(self, image):
        ''' version 2 of azimuthal averageing '''
        
        h  = image.shape[0]
        w  = image.shape[1]
        wc = w//2
        hc = h//2

        # create an array of integer radial distances from the center
        Y, X = np.ogrid[0:h, 0:w]
        r    = np.hypot(X - wc, Y - hc).astype(np.int)

        # SUM all psd2D pixels with label 'r' for 0<=r<=wc
        # NOTE: this will miss power contributions in 'corners' r>wc
        psd1D = ndimage.mean(image, r, index=np.arange(0, wc))

        return psd1D, len(psd1D)


    def calc_power_spec(self, time=[]):
        ''' calculate power_spectrum of 2d slice at depth '''


        # select time
        if time:
            model_slice = self.ds.isel(time_counter=time)
        else:
            model_slice = self.ds

        # get depth
        #depth_slice = model_slice.interp(deptht=10)
        depth_slice = model_slice.isel(deptht=10)

        # regrid and detrend
        self.interp_to_regular_grid()
        detrended = self.detrend()

        # windowing         
        tukey =  window(('tukey', 0.5), depth_slice.shape[1:])[None,:,:]
        depth_slice = depth_slice * tukey

        fourier = np.abs(fft.fftshift(fft.fft2(depth_slice.values))) ** 2
        time_dim_len = depth_slice.time_counter.size
        if not time:
            power_spec = []
            for time in range(time_dim_len):
                power_spec_t, nr = self.azimuthalAverage_2(fourier[time])
                power_spec.append(power_spec_t)
        else:
            power_spec, nr = self.azimuthalAverage_2(fourier)

        # get frequencies - this is probably incorrect
        #                   nr is number of radial bins
        #                   dx is mean zonal resolution
        dx = self.cfg.e1t.mean() # cell width mean
        freq = fft.fftshift(fft.fftfreq(nr, dx.values) * dx.values)
       
        return freq, np.array(power_spec)

#    def plot_multi_time_power_spectrum(self, times):
#
#        fig = plt.figure()
#        for time in times:
#            freq, power_spec = self.calc_2d_fft(time)
#            plt.loglog(freq, power_spec)
#
#        #plt.gca().set_xlim([1e-6,5e-5])
#        plt.gca().set_ylim([2e1,2e2])
#        plt.show()
    
if __name__ == '__main__':
    m = power_spectrum('EXP13', 'votemper')
    m.detrend()
    #m.plot_multi_time_power_spectrum(np.arange(0,100,10))
