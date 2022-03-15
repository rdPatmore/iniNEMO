import xarray as xr
import config
import gsw
import scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import window
from scipy.io.matlab import mio
import spectrum as sp
from scipy import stats
import itertools
import matplotlib.pyplot as plt

# not sure on the application here
# however the expectation is that, on average, slopes will reduce with increases
# in resolution as more submesoscale process are resolved.


class power_spectrum_model(object):

    def __init__(self, model, var):
        self.var = var 
        self.path = config.data_path() + model

    def get_model(self):
        if var in ['votemper', 'vosaline']:
            self.ds = xr.open_dataset(self.path + 
                            '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc',
                            chunks={'time_counter':10})
            self.ds = self.ds[var]
        self.cfg = xr.open_dataset(self.path + '/domain_cfg.nc')

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


    def calc_power_spec(self, time=[], load=True):
        ''' calculate power_spectrum of 2d slice at depth '''

        if load:
            detrended = xr.load_dataarray(self.path + '/Spectra/' +
                                           self.var + '_regular_grid_d10.nc')

        else: 
            # select time
            if time:
                self.ds = self.ds.isel(time_counter=time)

            self.ds = self.ds.isel(deptht=10)

            # regrid and detrend
            self.interp_to_regular_grid()
            detrended = self.detrend()

        # windowing         
        tukey =  window(('tukey', 0.5), detrended.shape[1:])[None,:,:]
        processed = detrended * tukey

        fourier = np.abs(fft.fftshift(fft.fft2(processed.values))) ** 2
        time_dim_len = processed.time_counter.size
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
        #freq = fft.fftfreq(nr, dx.values)
        #freq = fft.fftshift(fft.fftfreq(nr, dx.values) * dx.values)
        print (dx)
        freq = fft.fftshift(fft.fftfreq(nr, dx.values))*2*np.pi*1000
       
        return freq, np.array(power_spec)

    def prep_for_calc_power_spec(self, time=[]):
        ''' 
        prepares for calc_power_spec 
            - interpolates to regular grid
            - detrends data
            - saves a file for later loading
        '''

        # select time
        if time:
            self.ds = self.ds.isel(time_counter=time)

        self.ds = self.ds.isel(deptht=10)

        # regrid and detrend
        self.interp_to_regular_grid()
        detrended = self.detrend()

        detrended.name = self.var
        detrended.to_netcdf(self.path + '/Spectra/' +
                          self.var + '_regular_grid_d10.nc')

class power_spectrum_glider(object):

    def __init__(self, model, var, append='', fs=1000):
        self.var = var 
        self.append = append
        self.path = config.data_path() + model + '/'
        self.fs = fs

    def get_glider(self):
        ''' load process glider data '''

        def expand_sample_dim(ds):
            ds = ds.expand_dims('sample')
            ds['lon_offset'] = ds.attrs['lon_offset']
            ds['lat_offset'] = ds.attrs['lat_offset']
            ds = ds.set_coords(['lon_offset','lat_offset'])
            return ds
        file_paths = [self.path + 'GliderRandomSampling/glider_uniform_' +
                      self.append + str(i).zfill(2) + '.nc' for i in range(100)]
        self.glider = xr.open_mfdataset(file_paths,
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim,
                                         parallel=True)[self.var].load()
#    def get_transects(self, data):
#        a = np.abs(np.diff(data.lat, 
#           append=data.lon.max(), prepend=data.lon.min(), n=2))# < 0.001))[0]
#        idx = np.where(a>0.006)[0]
#        da = np.split(data, idx)
#        transect = np.arange(len(da))
#        for i, arr in enumerate(da):
#            da[i] = da[i].assign_coords({'transect':i})
#        da = xr.concat(da, dim='distance')
#        return da

    def get_transects(self, data, concat_dim='distance', method='cycle',
                      shrink=None):
        if method == '2nd grad':
            a = np.abs(np.diff(data.lat, 
            append=data.lon.max(), prepend=data.lon.min(), n=2))# < 0.001))[0]
            idx = np.where(a>0.006)[0]
        crit = [0,1,2,3]
        if method == 'cycle':
            #data = data.isel(distance=slice(0,400))
            data['orig_lon'] = data.lon - data.lon_offset
            data['orig_lat'] = data.lat - data.lat_offset
            idx=[]
            crit_iter = itertools.cycle(crit)
            start = True
            a = next(crit_iter)
            for i in range(data[concat_dim].size)[::shrink]:
                da = data.isel({concat_dim:i})
                if (a == 0) and (start == True):
                    test = ((da.orig_lat < -60.04) and (da.orig_lon > 0.176))
                elif a == 0:
                    test = (da.orig_lon > 0.176)
                elif a == 1:
                    test = (da.orig_lat > -59.93)
                elif a == 2:
                    test = (da.orig_lon < -0.173)
                elif a == 3:
                    test = (da.orig_lat > -59.93)
                if test: 
                    start = False
                    idx.append(i)
                    a = next(crit_iter)
        da = np.split(data, idx)
        transect = np.arange(len(da))
        pop_list=[]
        for i, arr in enumerate(da):
            if len(da[i]) < 1:
                pop_list.append(i) 
            else:
                da[i] = da[i].assign_coords({'transect':i})
        for i in pop_list:
            da.pop(i)
        da = xr.concat(da, dim=concat_dim)
        # remove initial and mid path excursions
        da = da.where(da.transect>1, drop=True)
        da = da.where(da.transect != da.lat.idxmin().transect, drop=True)
        return da


    def detrend(self, h, remove_mean=False):

        if remove_mean:
            h = h - h.mean()

        n = len(h)
        #print ('detrend length', n)
        t = np.arange(n)
        p = np.polyfit(t, h, 1)
        self.h_detrended = h - np.polyval(p, t)

    def multiTaper(self, x, dt=1., nw=3, nfft=None):
        """
        function [P,s,ci] = pmtmPH(x,dt,nw,qplot,nfft);
        Computes the power spectrum using the multi-taper method with adaptive 
        weighting.
        Inputs:
        x      - Input data vector.
        dt     - Sampling interval, default is 1.
        nw     - Time bandwidth product, acceptable values are
        0:.5:length(x)/2-1, default is 3.  2*nw-1 dpss tapers
        are applied except if nw=0 a boxcar window is applied 
        and if nw=.5 (or 1) a single dpss taper is applied.
        qplot  - Generate a plot: 1 for yes, else no.  
        nfft   - Number of frequencies to evaluate P at, default is
        length(x) for the two-sided transform. 
        Outputs:
        P      - Power spectrum computed via the multi-taper method.
        s      - Frequency vector.
        ci     - 95% confidence intervals. Note that both the degrees of freedom
        calculated by pmtm.m and chi2conf.m, which pmtm.m calls, are
        incorrect.  Here a quick approximation method is used to
        determine the chi-squared 95% confidence limits for v degrees
        of freedom.  The degrees of freedom are close to but no larger
        than (2*nw-1)*2; if the degrees of freedom are greater than
        roughly 30, the chi-squared distribution is close to Gaussian.
        The vertical ticks at the top of the plot indicate the size of
        the full band-width.  The distance between ticks would remain
        fixed in a linear plot.  For an accurate spectral estimate,
        the true spectra should not vary abruptly on scales less than
        the full-bandwidth.
        Other toolbox functions called: dpps.m; and if nfft does not equal 
        length(x)    , cz.m
        Peter Huybers
        MIT, 2003
        phuybers@mit.edu

        Adapted from Matlab to Python by Nicolas Barrier"""
        if nfft is None:
            nfft=len(x)

        nx=len(x)
        k=np.min([np.round(2*nw),nx])
        k=np.max([k-1,1])
        s=np.arange(0,1/dt,1/(nfft*dt));
        w=nw/(dt*nx) # half-bandwidth of the dpss
        
        E,V=sp.dpss(nx,NW=nw,k=k)
 
        if nx<=nfft:
            tempx=np.transpose(np.tile(x,(k,1)))
            Pk=np.abs(np.fft.fft(E*tempx,n=nfft,axis=0))**2
        else:
            raise IOError('Not implemented yet')
        
        #Iteration to determine adaptive weights:    
        if k>1:
            xmat=np.mat(x).T
            sig2 = xmat.T*xmat/nx; # power
            P    = (Pk[:,0]+Pk[:,1])/2.;   # initial spectrum estimate
            Ptemp= np.zeros(nfft);
            P1   = np.zeros(nfft);
            tol  = .0005*sig2/nfft;    
            a    = sig2*(1-V);
            #if  np.sum(np.abs(P-P1)/nfft)>tol:
            #    print ('win')
            #    # if first guess is already within tollerance
            #    Pmat=np.mat(P).T
            #    Vmat=np.mat(V)
            #    amat=np.mat(a)
            #    temp1=np.mat(np.ones((1,k)))
            #    temp2=np.mat(np.ones((nfft,1)))
            #    b=(Pmat*temp1)/(Pmat*Vmat+temp2*amat); # weights
            #    temp3=np.mat(np.ones((nfft,1)))*Vmat
            #    temp3=np.array(temp3)
            #    b=np.array(b)
            #    wk=b**2*temp3       
            #    P1=np.sum(wk*Pk,axis=1)/np.sum(wk,axis=1)
            #    Ptemp=P1; P1=P; P=Ptemp;                 # swap P and P1

            while np.sum(np.abs(P-P1)/nfft)>tol:
                Pmat=np.mat(P).T
                Vmat=np.mat(V)
                amat=np.mat(a)
                temp1=np.mat(np.ones((1,k)))
                temp2=np.mat(np.ones((nfft,1)))
                b=(Pmat*temp1)/(Pmat*Vmat+temp2*amat); # weights
                temp3=np.mat(np.ones((nfft,1)))*Vmat
                temp3=np.array(temp3)
                b=np.array(b)
                wk=b**2*temp3       
                P1=np.sum(wk*Pk,axis=1)/np.sum(wk,axis=1)
                Ptemp=P1; P1=P; P=Ptemp;                 # swap P and P1

            #b2=b**2
            #temp1=np.mat(np.ones((nfft,1)))*V
            temp1=b**2
            temp2=np.mat(np.ones((nfft,1)))*Vmat
            num=2*np.sum(temp1*np.array(temp2),axis=1)**2
            
            temp1=b**4
            temp2=np.mat(np.ones((nfft,1)))*np.mat(V**2)
            den=np.sum(temp1*np.array(temp2),axis=1)
            v=num/den
            
        select=np.arange(0,(nfft+1)/2+1).astype(np.int64)
        P=P[select]
        s=s[select]
        v=v[select]

        temp1=1/(1-2/(9*v)-1.96*np.sqrt(2./(9*v)))**3
        temp2=1/(1-2/(9*v)+1.96*np.sqrt(2/(9*v)))**3

        ci=np.array([temp1,temp2])

        return P,s,ci

    def glider_fft(self, signal):
        signal = signal.values
        N = len(signal)
        xdft = fft.fft(signal)
        xdft = xdft[:int(N/2)+1]
        #psdx = (self.fs/N) * np.abs(xdft)**2
        psdx = (self.fs/N**2) * np.abs(xdft)**2
        psdx[1:-1] = 2*psdx[1:-1]
        freq= np.linspace(0,1/(2*self.fs),len(psdx)) # if fs is set to distance
        return freq, psdx


    def glider_welch(self, signal):
        from scipy.signal import welch, hanning
        
        #freq= np.linspace(0,1/(2*self.fs),len(psdx)) # if fs is set to distance
        fs = 1/self.fs
        
        nblock = 300
        overlap = 20
        win = hanning(nblock, True)
        
        f, Pxxf = welch(signal, fs, window=win, noverlap=overlap, nfft=nblock,
                        return_onesided=True, detrend=False)
        
        return f, Pxxf
        
         

    def calc_spectrum(self, proc='multi_taper'):
        ''' 
        Calculate power spectrum with multi-taper method according to
        #Giddy 2020
        '''

        var10_stack = self.glider.sel(ctd_depth=10, method='nearest')
        
        #plt.figure()
        freq=np.linspace(0,1/(2*self.fs),4000) # if fs is set to distance
        #fs=np.linspace(0,0.5,1000) # if fs is set to grid number (i.e. 1)
        #fs=np.logspace(-8,0,1000)
        #for i in range(var10_stack.sample.size):
        mean_set = []
        ldecile_set = []
        udecile_set = []
        for i in range(var10_stack.sample.size):
        #for i in range(10):
            print ('sample: ', i)
            var10 = var10_stack.isel(sample=i).dropna(dim='distance')
            #var10 = self.get_transects(var10)
            #print (var10)
            Pset_transect = []
            for (label, transect) in var10.groupby('transect'):
                #print ('transect: ', label)
                if len(transect) <= 6:
                    continue
                self.detrend(transect, remove_mean=False)
                if proc == 'multi_taper':
                    if np.mean(self.h_detrended) == 0.0:
                        continue
                    PEm,sEm,ciEm = self.multiTaper(self.h_detrended, dt=self.fs)
                    PEm = 2 * PEm * self.fs# / (len(self.h_detrended)**2)
                if proc == 'fft':
                    sEm, PEm = self.glider_fft(self.h_detrended)
                if proc == 'welch':
                    sEm, PEm = self.glider_welch(self.h_detrended)
                PEm = np.interp(freq,sEm,PEm)
                Pset_transect.append(PEm) 
            Pset_transect_stack = np.stack(Pset_transect)

            mean_spec = np.nanmean(Pset_transect_stack, axis=0)
            ldecile_spec = np.quantile(Pset_transect_stack, 0.1, axis=0)
            udecile_spec = np.quantile(Pset_transect_stack, 0.9, axis=0)
            mean_set.append(mean_spec) 
            ldecile_set.append(ldecile_spec) 
            udecile_set.append(udecile_spec) 

        mean_Pset = np.stack(mean_set)
        ldecile_Pset = np.stack(ldecile_set)
        udecile_Pset = np.stack(udecile_set)

        ## overwrite freqencies with from cell metric to km metric
        #fs = fs / self.fs

        #print (mean_spec.shape)
        #plt.loglog(fs, mean_spec, lw=1.0, alpha=1, c='orange')
        #plt.show()
        ds = xr.Dataset(data_vars=dict(
                        temp_spec_mean=(['sample','freq'], mean_Pset),
                        temp_spec_l_decile=(['sample','freq'], ldecile_Pset),
                        temp_spec_u_decile=(['sample','freq'], udecile_Pset)),
                        coords=dict(freq=freq),
                        attrs=dict(description=self.var + 
                                     ' power spectrum for 100 glider samples'))
        if self.append == '':
            ds.to_netcdf(self.path + 'Spectra/glider_samples_' + self.var + 
                              '_spectrum' + self.append.rstrip('_') +
                        '_' + proc + '_pre_transect_clean_pfit1.nc')
        else:
            ds.to_netcdf(self.path + 'Spectra/glider_samples_' + self.var + 
                              '_spectrum_' + self.append.rstrip('_') +
                        '_' + proc + '_pre_transect_clean_pfit1.nc')

    def calc_variance(self, proc='fft'):
        ''' calculate integral under first sample spectrum '''
        
        spec = xr.open_dataset(self.path + 'Spectra/glider_samples_'
                              + self.var + '_spectrum' +
                              self.append.rstrip('_') + '_' + proc + '.nc')
        sample = spec.temp_spec.isel(sample=0)
        ds = sample.freq.diff(dim='freq')
        integ = (sample.isel(freq=slice(None,-1)) * ds).sum(dim='freq')
        print (integ.values)

if __name__ == '__main__':
    m = power_spectrum_glider('EXP10', 'votemper', 
                              append='burst_3_20_transects_',
                              fs=1000)
    m.get_glider()
    m.calc_spectrum(proc='multi_taper')
    #m = power_spectrum_glider('EXP10', 'votemper', 
    #                          append='every_2_',
    #                          fs=1000)
    #m.get_glider()
    #m.calc_spectrum(proc='multi_taper')
    #m = power_spectrum_glider('EXP10', 'votemper', 
    #                          append='every_8_and_climb_',
    #                          fs=1000)
    #m.get_glider()
    #m.calc_spectrum(proc='multi_taper')
    #m = power_spectrum_glider('EXP10', 'votemper', 
    #                          append='every_8_and_dive_',
    #                          fs=1000)
    #m.get_glider()
    #m.calc_spectrum(proc='multi_taper')

    #m = power_spectrum_glider('EXP10', 'votemper', 
    #                          append='interp_1000_',
    #                          fs=1000)
    #m.get_glider()
    #m.calc_spectrum(proc='multi_taper')
    #m = power_spectrum_glider('EXP10', 'votemper', 
    #                          append='interp_500_',
    #                          fs=500)
    #m.get_glider()
    #m.calc_spectrum(proc='multi_taper')
    #                          append='interp_2000_',
    #                          fs=2000)
    #m.get_glider()
    #m.calc_spectrum(proc='multi_taper')

    #for fs in [500,1000,2000]:
    #    if fs == 1000:
    #        m = power_spectrum_glider('EXP08', 'votemper', 
    #                              append='',
    #                              fs=fs)
    #    else:
    #        m = power_spectrum_glider('EXP08', 'votemper', 
    #                              append='_interp_'+str(fs)+'_',
    #                              fs=fs)
    #    m.calc_variance(proc='welch')
    #proc = 'multi_taper'
    #m = power_spectrum_glider('EXP08', 'votemper', append='interp_2000_',
    #                          fs=2000)
    #m.get_glider()
    #m.calc_spectrum(proc=proc)

    #m = power_spectrum_glider('EXP08', 'votemper', append='',
    #                          fs=1000)
    #m.get_glider()
    #m.calc_spectrum(proc=proc)

    #m = power_spectrum_glider('EXP08', 'votemper', append='interp_500_',
    #                          fs=500)
    #m.get_glider()
    #m.calc_spectrum(proc=proc)
