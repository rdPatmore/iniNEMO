import xarray as xr
import config
import gsw
import scipy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
import iniNEMO.Process.calc_power_spectrum as spec
from scipy import ndimage
import matplotlib
#from skimage.filters import window
import scipy.signal as sig
import matplotlib.gridspec as gridspec
from plot_interpolated_tracks import get_sampled_path, get_raw_path
import cartopy.crs as ccrs
import cartopy
import cartopy.mpl.geoaxes
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import haversine

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
        plt.subplots_adjust(left=0.12, bottom=0.1, right=0.96, top=0.98)
       
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

    def add_glider_spectra(self, model, ax, var='votemper', 
                           append='', c='orange',
                           label='', old=False, ls='-', old_spec_calc=False,
                           simple_calc=False, panel_label=None,
                           lines=False, zorder=1, a=0.4):
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
        #if c=='teal':
        #    spec['temp_spec_mean'] = spec.temp_spec_mean/2
        #    spec['temp_spec'] = spec.temp_spec/2
        #if append=='_interp_2000_multi_taper':
        #    spec['temp_spec_mean'] = spec.temp_spec_mean*2
        #    spec['temp_spec'] = spec.temp_spec*2
        #if append=='_multi_taper':
        #    spec['temp_spec_mean'] = spec.temp_spec_mean*2
        #    spec['temp_spec'] = spec.temp_spec*2
        #if append=='_interp_500_multi_taper':
        #    spec['temp_spec_mean'] = spec.temp_spec_mean*2
        #    spec['temp_spec'] = spec.temp_spec*2
        #    spec['temp_spec_mean'] = spec.temp_spec_mean*4
        #    spec['temp_spec'] = spec.temp_spec*4
        #print (spec)
        if old_spec_calc:
            spec_mean = spec.temp_spec_mean
            decile = spec.temp_spec.quantile([0.1,0.9], ['sample'])
            spec_l = decile.sel(quantile=0.1) 
            spec_u = decile.sel(quantile=0.9) 
        elif simple_calc:
            spec_mean = spec.temp_spec_gmean.mean(dim='sample', skipna=True)
            spec_l = spec.temp_spec_l_decile.mean(dim='sample', skipna=True)
            spec_u = spec.temp_spec_u_decile.mean(dim='sample', skipna=True)
        else:
            l_mag_mean = np.abs(spec.temp_spec_gmean - spec.temp_spec_l_decile
                               ).mean(dim='sample', skipna=True)
            u_mag_mean = np.abs(spec.temp_spec_gmean - spec.temp_spec_u_decile
                               ).mean(dim='sample', skipna=True)
            spec_mean = spec.temp_spec_gmean.mean(dim='sample', skipna=True)
            spec_l = spec_mean + u_mag_mean
            spec_u = spec_mean - l_mag_mean
        if lines:
            ax.loglog(spec_l.freq*1000, spec_l, c=c, lw=0.8, ls=':',
                      zorder=zorder)
            ax.loglog(spec_u.freq*1000, spec_u, c=c, lw=0.8, ls=':',
                      zorder=zorder)
        else:
            ax.fill_between(spec_l.freq*1000, spec_l, spec_u, alpha=a,
                             color=c, edgecolor=None, zorder=zorder)
        p = ax.loglog(spec_mean.freq*1000, spec_mean, c=c, alpha=1,
                       lw=1.2, label=label, ls=ls, zorder=zorder)
        if panel_label:
            ax.text(0.95, 0.97, panel_label, va='top', ha='right',
                    transform=ax.transAxes, fontsize=6)
        return p
    
    def finishing_touches(self):
        self.ax.set_xlabel(r'Wavenumber [km$^{-1}$]')
        self.ax.set_ylabel('Temperature Power Spectral Density')
        self.ax.set_ylim(1e-7,1e2)
        self.ax.set_xlim(2e-2,4)
        self.fig.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99),
                        fontsize=6)

    def get_raw_path(self):
        glider_raw = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
        glider_raw = glider_raw.rename({'longitude': 'lon', 'latitude': 'lat'})
        index = np.arange(glider_raw.ctd_data_point.size)
        glider_raw = glider_raw.assign_coords(ctd_data_point=index)
        self.glider_raw = get_transects(
                               glider_raw.dives,concat_dim='ctd_data_point',
                                   shrink=100, offset=False)

    def add_path(self, ax, glider_data, post_transect, path_cset):
        ''' add birdseye view of glider track '''

        # domain projection
        inset_proj=ccrs.AlbersEqualArea(central_latitude=-60,
                                    standard_parallels=(-62,-58))

        # set inset
        axins = inset_axes(ax, width='38%', height='38%',
                       loc='lower left',
                       axes_class=cartopy.mpl.geoaxes.GeoAxes, 
                       axes_kwargs=dict(map_projection=inset_proj))
        axins.spines['geo'].set_visible(False)
        axins.patch.set_alpha(0.0)

        # plot path
        proj = ccrs.PlateCarree() # lon lat projection
        glider_data = get_sampled_path('EXP10', glider_data,
                                post_transect=post_transect, drop_meso=True) 
        for i, (l,trans) in enumerate(glider_data.groupby('transect')):
            axins.plot(trans.lon, trans.lat, transform=proj, 
                       c=path_cset[int(trans.vertex[0])], lw=0.5)

    def plot_pre_post_transect_and_climb_dive_reduction(self):
        '''
        plot paths and associated spectra when retriving transects
        pre/post interpolation
        also additional row of removing climbs

        pt = post_transect for path
        '''
        # initialised figure
        fig, axs = plt.subplots(3, 4, figsize=(6.5, 5.5), dpi=300)
        plt.subplots_adjust(left=0.11,right=0.89,top=0.99,bottom=0.09,
                            hspace=0.05,wspace=0.05)

        # initialise class
        self.spec = plot_power_spectrum()

        # colours
        c0 = '#0d8ca7' # ref
        c1 = '#a7280d' # comparison
        path_cset=['k','#dad1d1', '#7e9aa5', '#55475a']
        path_cset=['#2fc300','#0091c3','#9400c3','#c33200']
        c0 = 'k'
        c1 = '#f18b00'
        path_cset=[c1,'navy','green','purple']

            
        # add full path spectrum to all panels
        for ax in axs.flatten():
            spec_append='_interp_1000_pre_transect_multi_taper_clean_pfit1'
            p0, = self.spec.add_glider_spectra('EXP10', ax, append=spec_append,
                                               c=c0, zorder=1, a=0.5)

        # ~~~ pre transect pairs ~~~ #

        spec_append = \
                   ['_interp_1000_every_2_pre_transect_multi_taper_clean_pfit1',
                    '_interp_1000_every_3_pre_transect_multi_taper_clean_pfit1',
                    '_interp_1000_every_4_pre_transect_multi_taper_clean_pfit1',
                    '_interp_1000_every_8_pre_transect_multi_taper_clean_pfit1']
        names = ['interp_1000_every_2_pre_transect',
                 'interp_1000_every_3_pre_transect',
                 'interp_1000_every_4_pre_transect',
                 'interp_1000_every_8_pre_transect']
        pl = ['every 2 pre-transect', 'every 3 pre-transect', 
              'every 4 pre-transect', 'every 8 pre-transect']

        for i in range(4):
            self.spec.add_glider_spectra('EXP10', axs[1,i], 
                        append=spec_append[i], panel_label=pl[i], c=c1, a=0.5)
            add_path(axs[1,i], names[i], post_transect=False)

        # ~~~ post transect pairs ~~~ #

        spec_append = \
                   ['_every_2_post_transect_multi_taper_clean_pfit1',
                    '_every_3_post_transect_multi_taper_clean_pfit1',
                    '_every_4_post_transect_multi_taper_clean_pfit1',
                    '_every_8_post_transect_multi_taper_clean_pfit1']
        names = ['every_2', 'every_3','every_4','every_8']
        pl = ['every 2 post-transect', 'every 3 post-transect',
              'every 4 post-transect', 'every 8 post-transect']

        for i in range(4):
            self.spec.add_glider_spectra('EXP10', axs[0,i], 
                        append=spec_append[i], panel_label=pl[i], c=c1, a=0.5)
            self.add_path(axs[0,i], names[i], post_transect=True)

        # ~~~ climb removal ~~~ #

        spec_append = \
        ['_interp_1000_every_2_and_climb_pre_transect_multi_taper_clean_pfit1',
         '_interp_1000_every_3_and_climb_pre_transect_multi_taper_clean_pfit1',
         '_interp_1000_every_4_and_climb_pre_transect_multi_taper_clean_pfit1']
        pl = ['every 2 and climb', 'every 3 and climb', 'every 4 and climb']

        for i in range(3):
            p1, = self.spec.add_glider_spectra('EXP10', axs[2,i], 
                        append=spec_append[i], panel_label=pl[i],
                        simple_calc=True, c=c1, a=0.4)

        # ~~~ add vertical path ~~~ #

        def get_zig_zag(ypos,xpos):
            p0dir = 7*np.pi/8.
            pt = (ypos, xpos)
            xpts = [pt[1]]
            ypts = [pt[0]]
            for i in range(8):
                pt = haversine.inverse_haversine(pt, 10.05, p0dir)
                xpts.append(pt[1])
                ypts.append(pt[0])
                pt = haversine.inverse_haversine(pt, 10.05, p0dir-(6*np.pi/8.))
                xpts.append(pt[1])
                ypts.append(pt[0])
            return np.array(xpts), np.array(ypts)

        # full path
        for ax in axs.flatten():
            xpts, ypts = get_zig_zag(0.9,0.4)
            ax.plot(xpts, ypts, lw=1.5, c=c0, transform=ax.transAxes)

        # every 2
        xpts, ypts = get_zig_zag(0.9,0.4)
        xpts = np.where(np.arange(len(xpts)) % 4 > 2, np.nan, xpts)
        ypts = np.where(np.arange(len(ypts)) % 4 > 2, np.nan, ypts)
        axs[0,0].plot(xpts, ypts, lw=1.5, c=c1, transform=axs[0,0].transAxes,
                      alpha=1.0)

        # every 3

        # set lims
        for ax in axs.flatten():
            ax.set_xlim(5e-2,1e0)
            ax.set_ylim(5e-5,3e1)

        # drop ticks 
        for ax in axs[:-1,:].flatten(): 
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten(): 
            ax.set_yticklabels([])

        # axis labels
        for ax in axs[-1,:]:
            ax.set_xlabel(r'Wavenumber [km$^{-1}$]')
        for ax in axs[:,0]:
            ax.set_ylabel('Temperature\nPower Spectral Density')

        axs[0,-1].legend([p0,p1], ['full', 'reduced'],
                         loc='upper left', bbox_to_anchor=(1.03, 1.0),
                         fontsize=6, title='Path', borderaxespad=0)

        # save
        plt.savefig('testing_proj_gmean_simple_calc_new_orange_base_purp.png')
        
    def plot_pair_remove_and_climb_dive_reduction(self):
        '''
        plot paths and associated spectra when removing dive-climb pairs
        also additional row of removing climbs
        '''
        # initialised figure
        fig, axs = plt.subplots(2, 4, figsize=(6.5, 3.5), dpi=300)
        plt.subplots_adjust(left=0.11,right=0.89,top=0.99,bottom=0.12,
                            hspace=0.05,wspace=0.05)

        # initialise class
        self.spec = plot_power_spectrum()

        # set colours
        #c1 = '#0d8ca7' # model mean colour
        #c0 = '#a7280d' # model mean colour
        c0 = 'k'
        c1 = '#f18b00'
        path_cset=[c1,'navy','lightseagreen','purple']

        # add full path spectrum to all panels
        for ax in axs.flatten():
            spec_append='_interp_1000_pre_transect_multi_taper_clean_pfit1'
            p0, = self.spec.add_glider_spectra('EXP10', ax, append=spec_append,
                                               c=c0, simple_calc=True, a=0.5)

        # ~~~ pre transect pairs ~~~ #

        spec_append = \
                   ['_interp_1000_every_2_pre_transect_multi_taper_clean_pfit1',
                    '_interp_1000_every_3_pre_transect_multi_taper_clean_pfit1',
                    '_interp_1000_every_4_pre_transect_multi_taper_clean_pfit1',
                    '_interp_1000_every_8_pre_transect_multi_taper_clean_pfit1']
        names = ['interp_1000_every_2_pre_transect',
                 'interp_1000_every_3_pre_transect',
                 'interp_1000_every_4_pre_transect',
                 'interp_1000_every_8_pre_transect']
        pl = ['every 2', 'every 3', 'every 4', 'every 8']

        for i in range(4):
            self.spec.add_glider_spectra('EXP10', axs[0,i], 
                        append=spec_append[i], panel_label=pl[i], c=c1,
                        simple_calc=True, a=0.5)
            self.add_path(axs[0,i], names[i], post_transect=True,
                          path_cset=path_cset)

        # ~~~ climb removal ~~~ #

        spec_append = \
        ['_interp_1000_every_2_and_climb_pre_transect_multi_taper_clean_pfit1',
         '_interp_1000_every_3_and_climb_pre_transect_multi_taper_clean_pfit1',
         '_interp_1000_every_4_and_climb_pre_transect_multi_taper_clean_pfit1']
        names = ['every_2', 'every_3','every_4','every_8']
        pl = ['every 2 and climb', 'every 3 and climb', 'every 4 and climb']

        for i in range(3):
            p1, = self.spec.add_glider_spectra('EXP10', axs[1,i], 
                        append=spec_append[i], panel_label=pl[i], c=c1,
                        simple_calc=True, a=0.5)
            self.add_path(axs[1,i], names[i], post_transect=True,
                          path_cset=path_cset)

        # ~~~ add vertical path ~~~ #

        def get_zig_zag(ypos,xpos):
            p0dir = 7*np.pi/8.
            pt = (ypos, xpos)
            xpts = [pt[1]]
            ypts = [pt[0]]
            for i in range(8):
                pt = haversine.inverse_haversine(pt, 10.05, p0dir)
                xpts.append(pt[1])
                ypts.append(pt[0])
                pt = haversine.inverse_haversine(pt, 10.05, p0dir-(6*np.pi/8.))
                xpts.append(pt[1])
                ypts.append(pt[0])
            return np.array(xpts), np.array(ypts)

        # full path
        for ax in axs.flatten():
            xpts, ypts = get_zig_zag(0.9,0.4)
            ax.plot(xpts, ypts, lw=1.5, c=c0, transform=ax.transAxes)

        # pair removal
        fac = [4,6,8,16]
        for i, ax in enumerate(axs[0]):
            # every 2
            xpts, ypts = get_zig_zag(0.9,0.4)
            xpts = np.where(np.arange(len(xpts)) % fac[i] > 2, np.nan, xpts)
            ypts = np.where(np.arange(len(ypts)) % fac[i] > 2, np.nan, ypts)
            ax.plot(xpts, ypts, lw=1.5, c=c1, transform=ax.transAxes)

        # pair removal
        fac = [4,6,8,16]
        for i, ax in enumerate(axs[1]):
            # every 2
            xpts, ypts = get_zig_zag(0.9,0.4)
            xpts = np.where(np.arange(len(xpts)) % fac[i] > 1, np.nan, xpts)
            ypts = np.where(np.arange(len(ypts)) % fac[i] > 1, np.nan, ypts)
            ax.plot(xpts, ypts, lw=1.5, c=c1, transform=ax.transAxes)

        # ~~~ formating ~~~ #

        # set lims
        for ax in axs.flatten():
            ax.set_xlim(5e-2,1e0)
            ax.set_ylim(5e-5,3e1)

        # drop ticks 
        for ax in axs[:-1,:].flatten(): 
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten(): 
            ax.set_yticklabels([])

        # axis labels
        for ax in axs[-1,:]:
            ax.set_xlabel(r'Wavenumber [km$^{-1}$]')
        for ax in axs[:,0]:
            ax.set_ylabel('Temperature\nPower Spectral Density')

        # letters
        letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        for i, ax in enumerate(axs.flatten()):
            ax.text(0.05, 0.92, letters[i], transform=ax.transAxes)

        axs[0,-1].legend([p0,p1], ['full', 'reduced'],
                         loc='upper left', bbox_to_anchor=(1.03, 1.0),
                         fontsize=6, title='Path', borderaxespad=0)

        # save
        plt.savefig(
     'pair_and_climb_remove_simple_calc_geo_mean_post_challenger.png', dpi=1200)

    def plot_pre_post_transect(self):
        '''
        plot paths and associated spectra when retriving transects
        pre/post interpolation
        '''

        # initialised figure
        fig = plt.figure(figsize=(5.5, 4.5), dpi=300)
        gs0 = gridspec.GridSpec(ncols=4, nrows=2)
        gs1 = gridspec.GridSpec(ncols=2, nrows=1)
        gs0.update(top=0.99, bottom=0.53, left=0.1,  right=0.98,
                   wspace=0.05, hspace=0.05)
        gs1.update(top=0.50, bottom=0.1, left=0.1,  right=0.98, wspace=0.05)

        axs0, axs1 = [], []
        #for i in range(3):
        #    for j in range(2):
                #axs0.append(fig.add_subplot(gs0[j,i], 
        for i in range(8):
                axs0.append(fig.add_subplot(gs0[i], 
                     projection=ccrs.AlbersEqualArea(central_latitude=60,
                      standard_parallels=(-62,-58)), frameon=False))
        for i in range(2):
            axs1.append(fig.add_subplot(gs1[i]))
        #axs1 = fig.add_subplot(gs1[0],
        #             projection=ccrs.AlbersEqualArea(central_latitude=60,
        #              standard_parallels=(-62,-58)))

        proj = ccrs.PlateCarree()


        # plot pre transect
        # full path
        full_path = get_sampled_path('EXP10', 'interp_1000_pre_transect',
                                     post_transect=False, drop_meso=True) 
        for (l,trans) in full_path.groupby('transect'):
            axs0[0].plot(trans.lon, trans.lat, transform=proj)

        # every other pair removed
        every_2 = get_sampled_path('EXP10', 'interp_1000_every_2_pre_transect',
                                     post_transect=False, drop_meso=True) 
        for (l,trans) in every_2.groupby('transect'):
            axs0[1].plot(trans.lon, trans.lat, transform=proj)

        # sample every 4 pairs
        every_4 = get_sampled_path('EXP10', 'interp_1000_every_4_pre_transect',
                                     post_transect=False, drop_meso=True) 
        for (l,trans) in every_4.groupby('transect'):
            axs0[2].plot(trans.lon, trans.lat, transform=proj)

        # sample every 4 pairs
        every_8 = get_sampled_path('EXP10', 'interp_1000_every_8_pre_transect',
                                     post_transect=False, drop_meso=True) 
        for (l,trans) in every_8.groupby('transect'):
            axs0[3].plot(trans.lon, trans.lat, transform=proj)

        # plot post transect
        # full path
        full_path = get_sampled_path('EXP10', 'interp_1000', drop_meso=True) 
        for (l,trans) in full_path.groupby('transect'):
            axs0[4].plot(trans.lon, trans.lat, transform=proj)

        # every other pair removed
        every_2 = get_sampled_path('EXP10', 'every_2', drop_meso=True) 
        for (l,trans) in every_2.groupby('transect'):
            axs0[5].plot(trans.lon, trans.lat, transform=proj)

        # sample every 4 pairs
        every_4 = get_sampled_path('EXP10', 'every_4', drop_meso=True) 
        for (l,trans) in every_4.groupby('transect'):
            axs0[6].plot(trans.lon, trans.lat, transform=proj)

        # sample every 8 pairs
        every_4 = get_sampled_path('EXP10', 'every_8', drop_meso=True) 
        for (l,trans) in every_4.groupby('transect'):
            axs0[7].plot(trans.lon, trans.lat, transform=proj)

        # plot spectra
        spec = plot_power_spectrum()
        
        spec.add_glider_spectra('EXP10', axs1[0], var='votemper',
                     append='_interp_1000_pre_transect_multi_taper_clean_pfit1',
                                c='orange',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        spec.add_glider_spectra('EXP10', axs1[0], var='votemper',
             append='_interp_1000_every_2_pre_transect_multi_taper_clean_pfit1',
                                c='green',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        spec.add_glider_spectra('EXP10', axs1[0], var='votemper',
             append='_interp_1000_every_4_pre_transect_multi_taper_clean_pfit1',
                                c='red',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        spec.add_glider_spectra('EXP10', axs1[0], var='votemper',
             append='_interp_1000_every_8_pre_transect_multi_taper_clean_pfit1',
                                c='blue',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)

        spec.add_glider_spectra('EXP10', axs1[1], var='votemper',
                    append='_interp_1000_post_transect_multi_taper_clean_pfit1',
                                c='orange',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        spec.add_glider_spectra('EXP10', axs1[1], var='votemper',
                        append='_every_2_post_transect_multi_taper_clean_pfit1',
                                c='green',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        spec.add_glider_spectra('EXP10', axs1[1], var='votemper',
                        append='_every_4_post_transect_multi_taper_clean_pfit1',
                                c='red',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        spec.add_glider_spectra('EXP10', axs1[1], var='votemper',
                        append='_every_8_post_transect_multi_taper_clean_pfit1',
                                c='blue',
                                label='', old=False, ls='-', 
                                old_spec_calc=False,
                                simple_calc=False)
        for ax in axs1:
            ax.set_xlim(5e-2,5e-1)

        axs1[0].set_xlabel(r'Wavenumber [km$^{-1}$]')
        axs1[1].set_xlabel(r'Wavenumber [km$^{-1}$]')
        axs1[0].set_ylabel('Temperature Power Spectral Density')
        axs1[1].set_yticklabels([])
        #axs0.set_yticks([])
        #axs0.set_xticks([])
        #for ax in axs0:
        #    for side in ['left','right','top','bottom']:
        #        print (side)
        #        ax.spines[side].set_visible(False)
        plt.savefig('EXP10_transect_pre_post_alt.png', dpi=600)

    def compare_climb_dive_pair_reduction(self):
        ''' 
        multi-panel plot of removing:
              (upper) every 2, 3, 4 and 8 pairs 
              (lower) up, down, 2 and up, 2 and down 

        !!!This should be pre-interpolation transecting!!!

        '''
    
        fig, axs = plt.subplots(2, 4, figsize=(6.5,4))
        plt.subplots_adjust(right=0.88)
    
        spec = plot_power_spectrum()
        
        # commom plotting parameters
        simple_calc = False
        old = False
        old_spec_calc = True
        label = ''

        for ax in axs.flatten():
            spec.add_glider_spectra('EXP10', ax, var='votemper',
                     append='_interp_1000_pre_transect_multi_taper_clean_pfit1',
                                    c='green',
                                    label='', old=old, ls='-', 
                                    old_spec_calc=old_spec_calc,
                                    simple_calc=simple_calc)

        # remove every 2
        spec.add_glider_spectra('EXP10', axs[0,0], var='votemper',
         append='_interp_1000_every_2_pre_transect_multi_taper_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)
    
        # remove every 3
        spec.add_glider_spectra('EXP10', axs[0,1], var='votemper',
         append='_interp_1000_every_3_pre_transect_multi_taper_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)

        # remove every 4
        spec.add_glider_spectra('EXP10', axs[0,2], var='votemper',
         append='_interp_1000_every_4_pre_transect_multi_taper_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)

        # remove every 8
        spec.add_glider_spectra('EXP10', axs[0,3], var='votemper',
         append='_interp_1000_every_8_pre_transect_multi_taper_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)

        # remove climbs
        #spec.add_glider_spectra('EXP10', axs[1,0], var='votemper',
        # append='_climb_multi_taper_transect_clean_pfit1',
        #                            c='orange',
        #                            label='', old=False, ls='-', 
        #                            old_spec_calc=False,
        #                            simple_calc=False)

        # remove every 2 and climb
        spec.add_glider_spectra('EXP10', axs[1,0], var='votemper',
       append='_interp_1000_every_2_and_climb_multi_taper_transect_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)
        # remove every 3 and climb
        spec.add_glider_spectra('EXP10', axs[1,1], var='votemper',
       append='_interp_1000_every_3_and_climb_multi_taper_transect_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)

        # remove every 2 and climb
        spec.add_glider_spectra('EXP10', axs[1,2], var='votemper',
   append='_interp_1000_every_4_and_climb_pre_transect_multi_taper_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)

        # remove every 8 and climb
        spec.add_glider_spectra('EXP10', axs[1,3], var='votemper',
   append='_interp_1000_every_8_and_climb_pre_transect_multi_taper_clean_pfit1',
                                    c='orange',
                                    label='', old=False, ls='-', 
                                    old_spec_calc=False,
                                    simple_calc=simple_calc)
        for ax in axs.flatten():
            ax.set_xlim(5e-2,5e-1)
        for ax in axs[:,0]:
            ax.set_ylabel('Temperature\nPower Spectral Density')
        for ax in axs[1]:
            ax.set_xlabel(r'Wavenumber [km$^{-1}$]')
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        for ax in axs[0]:
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.NullFormatter())
            #ax.set_xticks([1e-1])
            #ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
            #ax.set_xticks([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        plt.savefig('EXP10_multi_panel_sampling_reductions.png', dpi=600)
    
    
          
def glider_sampling_alteration():
    m = plot_power_spectrum()
    m.ini_figure()
    m.add_glider_spectra('EXP10', 
                         append='_interp_1000_multi_taper_transect_clean_pfit1',
                         c='limegreen', label='full path')
    m.add_glider_spectra('EXP10', 
                         append='_burst_9_21_transects_multi_taper_pre_transect_clean_pfit1',
                         c='navy', label='burst 9-21 pre-t')
    m.add_glider_spectra('EXP10', 
                         append='_burst_3_20_transects_multi_taper_pre_transect_clean_pfit1',
                         c='orange', label='burst 3-21 pre-t')
    m.add_glider_spectra('EXP10', 
                         append='_burst_3_9_transects_multi_taper_pre_transect_clean_pfit1',
                         c='red', label='burst 3-9 pre-t')
    #m.add_glider_spectra('EXP08', append='_interp_500_multi_taper', c='teal',
    #                     label='interval 500 m multi-taper')

    #m.add_glider_spectra('EXP08', append='_interp_2000_fft', c='blue', ls='--',
    #                     label='interval 2000 m fft')
    #m.add_glider_spectra('EXP08', append='_fft', c='orange', ls='--',
    #                     label='interval 1000 m fft')
    #m.add_glider_spectra('EXP08', append='_interp_500_fft', c='teal', ls='--',
    #                     label='interval 500 m fft')
    #m.add_glider_spectra('EXP08', append='_interp_2000_welch', c='blue',
    #                     ls=':', label='interval 2000 m welch')
    #m.add_glider_spectra('EXP08', append='_welch', c='orange',
    #                     ls=':', label='interval 1000 m welch')
    #m.add_glider_spectra('EXP08', append='_interp_500_welch', c='teal', ls=':',
    #                     label='interval 500 m welch')
    #m.add_glider_spectra('EXP02', c='orange', label='full path', old=True)
    #m.add_glider_spectra('EXP02', append='_climb', c='teal', label='no climb',
    #                     old=True)
    m.finishing_touches()
    #plt.show()
    plt.savefig('EXP10_glider_burst_3_21_9_21_3_9_transect_clean_polyfit1.png',
                dpi=1200)
#glider_sampling_alteration()
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

m = plot_power_spectrum()
#m.plot_pre_post_transect()
#m.plot_pre_post_transect_and_climb_dive_reduction()
m.plot_pair_remove_and_climb_dive_reduction()

#m.compare_climb_dive_pair_reduction()
#m.plot_regridded_detrended_example()
