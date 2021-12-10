import xarray as xr
import config
import gsw
import scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
import iniNEMO.Process.calc_power_spectrum as spec
from scipy import ndimage
import matplotlib
from skimage.filters import window
import scipy.signal as sig

matplotlib.rcParams.update({'font.size': 8})

# not sure on the application here
# however the expectation is that, on average, slopes will reduce with increases
# in resolution as more submesoscale process are resolved.


class plot_power_spectrum(object):

    def __init__(self):
        print ('start')
        self.domain_lim = 1000
        self.data_path_old = config.data_path_old()
        self.data_path = config.data_path()

    def toy_signal(self):
        ''' idealised signal for testing spectra code '''
        
  
        def calc_spec(z):
            #z = z * window(('tukey', 0.1), z.shape)
            fourier = np.abs(fft.fftshift(fft.fft2(z)))# ** 2

            h  = fourier.shape[0]
            w  = fourier.shape[1]
            wc = w//2
            hc = h//2

            # create an array of integer radial distances from the center
            Y, X = np.ogrid[0:h, 0:w]
            r    = np.hypot(X - wc, Y - hc).astype(np.int64)

            # SUM all psd2D pixels with label 'r' for 0<=r<=wc
            # NOTE: this will miss power contributions in 'corners' r>wc
            psd1D = ndimage.mean(fourier, r, index=np.arange(0, wc))
            r = np.sin(np.linspace(0, np.pi/2, wc))
            psd1D = psd1D*r#[::-1]
            return psd1D

        def dummy_polar(freq):

            r = np.linspace(0, 2*np.pi, self.domain_lim)
            p = np.linspace(0, 2*np.pi, self.domain_lim)
            R, P = np.meshgrid(r, p)
            #Z = ((R**2 - 1)**2)
            Z = -np.cos(R*freq)
            
            # Express the mesh in the cartesian system.
            X, Y = R*np.cos(P), R*np.sin(P)
            return (X,Y,Z)

        def add_1d(freq, c):
            x = np.linspace(0, 2*np.pi, self.domain_lim)
            y = -np.cos(x*freq)
            #y = y * sig.tukey(self.domain_lim, alpha=0.1)
            #fourier = np.abs(fft.fftshift(fft.rfft(y))) ** 2
            fourier = np.abs((fft.fft(y)))# ** 2
            print (len(fourier))
            r = np.linspace(0, 2*np.pi, self.domain_lim)
            plt.loglog(r, fourier, label=str(freq) + '_1d', c=c, ls='--')
            #plt.plot(y, label=str(freq) + '_1d')

        plt.figure(0)

        freqs = [1,2,3,5,10,20]
        colours = plt.cm.plasma(np.linspace(0,1,6))
        for i, freq in enumerate(freqs):
            z = calc_spec(dummy_polar(freq)[2])
            #r = np.linspace(0, np.pi, int(self.domain_lim/2))
            r = np.linspace(0, np.pi, int(z.shape[0]))
            plt.loglog(r,z, label=str(freq), c=colours[i])
            add_1d(freq, c=colours[i])
        plt.gca().set_xlim([5e-3,np.pi])
        plt.gca().set_ylim([1e-7,1e7])
        #plt.legend()
        #plt.show()
        plt.savefig('test1.png')
        
        def plot_3d(fig, num, freq):
            ax = fig.add_subplot(2,3,num, projection='3d')
            x,y,z = dummy_polar(freq)
            ax.plot_surface(x,y,z, color='darkcyan', 
                            vmin=np.nanmin(z), vmax=np.nanmax(z), shade=True,
                            antialiased=True, alpha=1.0, rstride=1, cstride=1,
                            linewidth=0)
            ax.set_zlim(-1,11)
            ax.grid(False)
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.set_facecolor('none')
            ax.set_xticks([-2*np.pi, 0, 2*np.pi])
            ax.set_yticks([-2*np.pi, 0, 2*np.pi])
            ax.set_xticklabels([r'2$\pi$', 0, r'2$\pi$'])
            ax.set_yticklabels([r'2$\pi$', 0, r'2$\pi$'])
            ax.set_zticks([])
            ax.w_zaxis.line.set_lw(0.)

        def plot_2d(fig, num, freq):
            ax = fig.add_subplot(2,3,num)
            x,y,z = dummy_polar(freq)
            ax.pcolor(x,y,z, cmap=plt.cm.inferno,vmin=np.min(z), vmax=np.max(z))

        #fig = plt.figure(1, figsize=(6.5,3))
        #for i, freq in enumerate(freq):
        #    plot_3d(fig, i+1, freq)
        #plt.subplots_adjust(top=1.3,right=0.95, left=0.01, 
        #                    hspace=-0.3, wspace=0.00)
        #plt.savefig('test2.png')

        #fig = plt.figure(1, figsize=(6.5,4))
        #for i, freq in enumerate(freq):
        #    plot_2d(fig, i+1, freq)
        #plt.savefig('test3.png')

    def plot_regridded_detrended_example(self):
        '''
        3 panel plot of one time step and depth level that
        has been
            - regridded to regular grid
            - detrended to remove large scale variability
        panel 1: original data
              2: original minus trend
              3: trend
        '''

        fig, axs = plt.subplots(1,3)
        plt.subplots_adjust(bottom=0.2)

        model_spec = spec.power_spectrum('EXP13', 'votemper')
        model_spec.ds = model_spec.ds.isel(deptht=10, time_counter=0)

        model_spec.interp_to_regular_grid()
        detrended = model_spec.detrend()

        trend = model_spec.ds - detrended

        x = model_spec.ds.x
        y = model_spec.ds.y

        p0 = axs[0].pcolor(x, y, model_spec.ds, vmin=-2, vmax=0.5)
        axs[1].pcolor(x, y, detrended, vmin=-2, vmax=0.5)
        p2 = axs[2].pcolor(x, y, trend, vmin=0, vmax=1.5)
         
        def add_colourbar(ax, p):
            pos = ax.get_position()
            cbar_ax = fig.add_axes([pos.x0, 0.1, pos.x1 - pos.x0, 0.02])
            cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
            cbar.ax.text(0.5, -3.5, r'Temperature ($^{\circ}$C)', fontsize=8,
              rotation=0, transform=cbar.ax.transAxes, va='bottom', ha='center')

        add_colourbar(axs[0], p0)
        add_colourbar(axs[2], p2)

        for ax in axs:
            ax.set_aspect('equal')
        plt.show()
         
    def plot_multi_time_power_spectrum(self, times):

        fig = plt.figure(figsize=(3.2,4))
        model_spec = spec.power_spectrum('EXP13', 'votemper')
        for time in times:
            freq, power_spec = model_spec.calc_2d_fft(time)
            plt.loglog(freq, power_spec)

        #plt.gca().set_xlim([1e-6,5e-5])
        plt.gca().set_ylim([2e1,2e2])
        plt.show()

    def get_power_spec_stats(self):
        model_spec = spec.power_spectrum('EXP13', 'votemper')
        print (model_spec.ds.time_counter.size)
        freq, power_spec = model_spec.calc_power_spec()
        print (power_spec)
        #for time in times:
        #    freq, power_spec = model_spec.calc_2d_fft(time)
        #mean_spec = np.mean(power_spec, axis=0)
        #std_spec = np.std(power_spec, axis=0)
        #lower = mean_spec - 2 * std_spec
        #upper = mean_spec + 2 * std_spec
        (lower, middle, upper) = np.quantile(power_spec, [0.1,0.5,0.9], axis=0)
        
        #plt.loglog(freq, power_spec.T, color='blue', alpha=0.01)
        plt.fill_between(freq, lower, upper, alpha=0.2)
        plt.loglog(freq, middle)

        #plt.gca().set_xlim([1e-6,5e-5])
        #plt.gca().set_ylim([8e-1,2e2])
        plt.gca().set_ylim([1e-6,1e9])
        plt.show()

    def get_power_spec_stats_multi_model(self, models, labels):
        c = ['red','blue']
        fig, self.ax = plt.subplots(1,1,figsize=(4.5,4.5))
        plt.subplots_adjust(left=0.12, bottom=0.1, right=0.98, top=0.98)
       
        for i, model in enumerate(models):
            model_spec = spec.power_spectrum(model, 'votemper')
            freq, power_spec = model_spec.calc_power_spec()
            (lower, middle, upper) = np.quantile(power_spec, [0.1,0.5,0.9],
                                                 axis=0)
            
            self.ax.fill_between(freq, lower, upper, alpha=0.2, color=c[i])
            self.ax.loglog(freq, middle, color=c[i], label=labels[i])

            self.ax.set_ylim([1e-2,2e0])
            self.ax.set_ylim([1e-1,5e2])

    def add_power_law(self, power, ls='-'):
        ''' adds Kolmogorov-esk power law to spectra plot ''' 
      
        try:
            k = float(power)
        except:
            num, denom = power.split('/') 
            k = float(num) / float(denom)

        y0 = 1e2 # start y pos of line

        x = np.array([3.6e-1,1.05-0])
        ystage=np.log(y0)-k*(np.log(x[0])-np.log(x))
        y=np.exp(ystage)

        #x = np.array([1e-1,1.5e-0])
        #c = y0 - (x[0] ** (power))
        #y = c + (x ** (power))
        #print (c)
        #print (x)
        #print (y)

        self.ax.loglog(x, y, ls=ls, color='grey')
        plt.text(x[-1],y[-1],' k='+power, ha='left',
                va='center', color='grey',fontsize='8')

    def format_axes(self):
        ax = plt.gca()
        ax.set_xlabel(r'Wavenumber (km$^{-1}$)')
        ax.set_ylabel('Temperature Power Spectral Density')
        ax.legend()


    def ini_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(3.5,3.5))
        plt.subplots_adjust(left=0.15, top=0.98, right=0.98)

    def add_glider_spectra(self, model, var='votemper', append='', c='orange',
                           label='', old=False):
        ''' plot glider spectrum with alterations'''

        # get spectrum
        if old:
            path = self.data_path_old + model + '/Spec/'
        else:
            path = self.data_path + model + '/Spectra/'
        spec = xr.load_dataset(path + 'glider_samples_' + var + 
                               '_spectrum' + append + '.nc')
        #for group, samples in spec.groupby('sample'):
        #    print (group)
        #    self.ax.loglog(samples.freq, samples.temp_spec, c=c,
        #                   alpha=0.01, lw=0.5)
        decile = spec.temp_spec.quantile([0.1,0.9], ['sample'])
        spec_l = decile.sel(quantile=0.1) 
        spec_u = decile.sel(quantile=0.9) 
        self.ax.fill_between(spec_l.freq, spec_l, spec_u, alpha=0.2, color=c,
                             edgecolor=None)
        self.ax.loglog(spec.freq, spec.temp_spec_mean, c=c, alpha=1, lw=0.8,
                       label=label)
    
    def finishing_touches(self):
        self.ax.set_xlabel(r'Wavenumber [km$^{-1}$]')
        self.ax.set_ylabel('Temperature Power Spectral Density')
        self.ax.set_ylim(1e-9,1e2)
        self.fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
def glider_sampling_alteration():
    m = plot_power_spectrum()
    m.ini_figure()
    #m.add_glider_spectra('EXP08', c='orange', label='full path')
    #m.add_glider_spectra('EXP08', append='_climb', c='teal', label='no climb')
    m.add_glider_spectra('EXP02', c='orange', label='full path', old=True)
    #m.add_glider_spectra('EXP02', append='_climb', c='teal', label='no climb',
    #                     old=True)
    m.finishing_touches()
    plt.savefig('EXP02_glider_spectra_full_only.png', dpi=1200)
glider_sampling_alteration()
##m.toy_signal()
##m.plot_multi_time_power_spectrum(np.arange(0,100,10))
def model_res_compare():
    m = plot_power_spectrum()
    m.get_power_spec_stats_multi_model(['EXP13','EXP08'],
                                       [r'1/12$^{\circ}$',r'1/24$^{\circ}$'])
    m.format_axes()
    m.add_power_law('-5/3')
    m.add_power_law('-2', ls='--')
    m.add_power_law('-3', ls=':')
    plt.savefig('temperature_spectra_method2_completeness.png', dpi=600)

#m = plot_power_spectrum()
#m.plot_regridded_detrended_example()
