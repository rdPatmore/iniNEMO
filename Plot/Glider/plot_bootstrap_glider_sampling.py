import xarray as xr
import config
#import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
import dask
import matplotlib
import datetime
import matplotlib.gridspec as gridspec
import scipy.stats as stats
#import itertools
from iniNEMO.Process.Glider.get_transects import get_transects

#matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 8})
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

class bootstrap_plotting(object):
    def __init__(self, append='', bg_method='norm', interp='1000'):
        self.data_path = config.data_path()
        if append == '':
            self.append = ''
        else:
            self.append='_' + append

        self.interp = '_interp_' + interp

        self.hist_range = (0,2e-8)
        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.bg_method = bg_method

    def plot_variance(self, cases):

        fig, axs = plt.subplots(3,2, figsize=(6.5,5.5))

        def render(ax, g_ens, c='black', var='mean'):
            pre = 'b_x_ml_' + var
            if var == 'mean':
                x = 'time_counter_mean'
            else:
                x = 'day'
            ax.fill_between(g_ens[x],
                            g_ens[pre + '_set_quant'].sel(quantile=0.05),
                            g_ens[pre + '_set_quant'].sel(quantile=0.95),
                            color=c, alpha=0.2)
            ax.plot(g_ens[x], g_ens[pre + '_set_mean'], c=c)
            ax.set_xlim(g_ens[x].min(skipna=True),
                        g_ens.dropna('day')[x].max(skipna=True))

        for i, case in enumerate(cases): 

            # get data
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            g = xr.open_dataset(path + prepend +  'glider_timeseries' +
                                self.append + '.nc')
            m_std = xr.open_dataset(path + prepend + 'day_week_std_timeseries' +
                                 self.append + '.nc')
            m_mean = xr.open_dataset(path + prepend + 'stats_timeseries' + 
                                 self.append + '.nc')

            # convert to datetime
            g['time_counter_mean'] = g.time_counter_mean.astype(
                                                               'datetime64[ns]')
            #g = g.dropna('day')
            #m_std = m_std.dropna('day')
            #g['b_x_ml_week_std_set_quant'] = g.b_x_ml_week_std_set_quant.dropna('day')

            # model - sdt
            axs[i,1].plot(m_std.day, m_std.bx_ts_day_std, c='cyan')
            axs[i,1].plot(m_std.day, m_std.by_ts_day_std, c='cyan', ls=':')

            # model - mean
            axs[i,0].plot(m_mean.time_counter, m_mean.bx_ts_mean, c='cyan')
            axs[i,0].plot(m_mean.time_counter, m_mean.by_ts_mean, c='cyan',
                                                                    ls=':')
            
            # 1 glider
            g1 = g.isel(ensemble_size=0)
            render(axs[i,1], g1, c='green', var='day_std')
            render(axs[i,0], g1, c='green', var='mean')

            # 4 glider
            g1 = g.isel(ensemble_size=3)
            render(axs[i,1], g1, c='red', var='day_std')
            render(axs[i,0], g1, c='red', var='mean')

            # 30 glider
            g1 = g.isel(ensemble_size=29)
            render(axs[i,1], g1, c='navy', var='day_std')
            render(axs[i,0], g1, c='navy', var='mean')

        for ax in axs[:,0]:
            ax.set_ylim(0,2e-7)
        for ax in axs[:,1]:
            ax.set_ylim(0,1e-7)

        plt.show()

    def render_glider_sample_set_v(self, n=1, c='green', style='plot',
                                   by_time=None):
        # weekly glider time series
        ds = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                           str(n).zfill(2) + '_hist' + self.append 
                          + '_' + by_time + self.interp + '.nc')

        # entier glider time series
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                          str(n).zfill(2) + '_hist' + self.append + '.nc')

        # regular weeks
        date_list = np.array([(np.datetime64('2012-12-13')
                              + np.timedelta64(i, 'W')).astype('datetime64[D]')
                               for i in range(16)])
        if style=='bar':
            for (l, week) in ds.groupby('time_counter'):
                i = int(np.argwhere(date_list==l.astype('datetime64[D]')))
                self.axs.flatten()[i].barh(week.bin_left, 
                                 week.hist_u_dec - week.hist_l_dec, 
                                 height=week.bin_right - week.bin_left,
                                 color=c,
                                 alpha=1.0,
                                 left=week.hist_l_dec, 
                                 align='edge',
                                 label='gliders: ' + str(n))
                self.axs.flatten()[i].text(0.3, 0.9, 
                                  week.time_counter.dt.strftime('%m-%d').values,
                                  transform=self.axs.flatten()[i].transAxes,
                                  fontsize=6)
                self.axs.flatten()[i].text(0.3, 0.8, 
                                  str(week.sample_size.values),
                                  transform=self.axs.flatten()[i].transAxes,
                                  fontsize=6)

            self.axs[1,-1].barh(ds_all.bin_left, 
                             ds_all.hist_u_dec - ds_all.hist_l_dec, 
                             height=ds_all.bin_right - ds_all.bin_left,
                             color=c,
                             alpha=1.0,
                             left=ds_all.hist_l_dec, 
                             align='edge',
                             label='gliders: ' + str(n))
            self.axs[1,-1].text(0.3, 0.9, 'all',
                              transform=self.axs[1,-1].transAxes,
                              fontsize=6)
            #    self.ax.scatter(ds.bin_centers, ds.hist_mean, c=c, s=4, zorder=10)
            if style=='plot':
                self.ax.fill_between(ds.bin_centers, ds.hist_l_dec,
                                                     ds.hist_u_dec,
                                     color=c, edgecolor=None, alpha=0.2)
                self.ax.plot(ds.bin_centers, ds.hist_mean, c=c, lw=0.8,
                             label='gliders: ' + str(n))

    def add_model_means_v(self, style='plot', by_time=None):
        ''' add model means of the normed buoyancy gradients of the model '''
        ds = xr.open_dataset(self.path + 
                           '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                           self.append + '_' + by_time + '.nc')
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                        self.append + '.nc')
        # mean direction
        if self.bg_method == 'mean':
            ds['hist'] = (ds.hist_x + ds.hist_y) / 2
            ds_all['hist'] = (ds_all.hist_x + ds_all.hist_y) / 2

        # vector norm
        if self.bg_method == 'norm':
            ds['hist'] = ds.hist_norm
            ds_all['hist'] = ds_all.hist_norm

        date_list = np.array([(np.datetime64('2012-12-13')
                              + np.timedelta64(i, 'W')).astype('datetime64[D]')
                               for i in range(16)])
        if style=='bar':
            for (l, week) in ds.groupby('time_counter'):
                i = int(np.argwhere(date_list==l.astype('datetime64[D]')))
                self.axs.flatten()[i].vlines(week.hist,
                        week.bin_left, week.bin_right,
                       transform=self.axs.flatten()[i].transData,
                       colors='k', lw=0.8, label='model bgx')
            self.axs[1,-1].vlines(ds_all.hist,
                       ds_all.bin_left, ds_all.bin_right,
                       transform=self.axs[1,-1].transData,
                       colors='k', lw=0.8, label='model bgx')
        if style=='plot':
            self.ax.plot(ds.bin_centers, ds.hist, c='black', lw=0.8,
                         label='model bg')

    def add_model_means_averaged_over_weeks(self, axs, c='k'):
        ''' 
        add model means of normed buoyancy gradients of model averaged over
        time slots
        '''
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                        self.append + '.nc')
        # mean direction
        if self.bg_method == 'mean':
            ds_all['hist'] = (ds_all.hist_x + ds_all.hist_y) / 2

        # vector norm
        if self.bg_method == 'norm':
            ds_all['hist'] = ds_all.hist_norm

        rolls = ['1W_rolling','2W_rolling','3W_rolling']
        for i, roll_freq in enumerate(rolls):
            ds = xr.open_dataset(self.path + 
                           '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                           self.append + '_' + roll_freq + '.nc')
            ds = ds.mean('time_counter')

            # mean direction
            if self.bg_method == 'mean':
                ds['hist'] = (ds.hist_x + ds.hist_y) / 2
      
            # vector norm
            if self.bg_method == 'norm':
                ds['hist'] = ds.hist_norm

            axs.flatten()[i].vlines(ds.hist,
                   ds.bin_left, ds.bin_right,
                   transform=axs.flatten()[i].transData,
                   colors=c, lw=0.8, label='model')
        l = axs[-1].vlines(ds_all.hist,
                   ds_all.bin_left, ds_all.bin_right,
                   transform=axs[-1].transData,
                       colors=c, lw=0.8, label='model')
        return l

    def render_glider_sample_set_averaged_over_weeks(self, axs, sample_sizes,
                                                     c='green'):

        l = []
        for i, n in enumerate(sample_sizes):
            print ('sample', i)
            ds_rolling = self.ds_rolling.isel(glider_quantity=n)
            ds_rolling = ds_rolling.mean('time_counter')
            for j, (_, ds) in enumerate(ds_rolling.groupby('rolling')):
                # weekly glider time series
                axs.flatten()[j].barh(ds.bin_left, 
                                 ds.hist_u_dec - ds.hist_l_dec, 
                                 height=ds.bin_right - ds.bin_left,
                                 color=c[i],
                                 alpha=1.0,
                                 left=ds.hist_l_dec, 
                                 align='edge',
                                 label='gliders: ' + str(n))

            ds_all = self.ds_all.isel(glider_quantity=n)
            l.append(axs[-1].barh(ds_all.bin_left, 
                             ds_all.hist_u_dec - ds_all.hist_l_dec, 
                             height=ds_all.bin_right - ds_all.bin_left,
                             color=c[i],
                             alpha=1.0,
                             left=ds_all.hist_l_dec, 
                             align='edge',
                             label='gliders: ' + str(n)))
        return l

    def add_giddy(self, by_time=None):
        ''' add giddy buoyancy gradient distribution '''

        # get glider data
        root = config.root()
        giddy = xr.open_dataset(root + 'Giddy_2020/sg643_linterp.nc')
        giddy_10 = giddy.sel(depth=10, method='nearest')
        giddy_10 = giddy_10.set_coords('time')

        # calculate buoyancy gradients
        g = 9.81     # gravity 
        rho_0 = 1026 # reference density
        b = g*(1-giddy_10.dens/rho_0)
        dx = 1000
        dbdx = b.diff('distance') / dx

        def get_hist(bx):
            hist, bins = np.histogram(bx.dropna('time', how='all'),
                                  range=self.hist_range, density=True, bins=20)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # assign to dataset
            hist_ds = xr.Dataset({'hist':(['bin_centers'], hist)},
                       coords={'bin_centers': (['bin_centers'], bin_centers),
                               'bin_left'   : (['bin_centers'], bins[:-1]),
                               'bin_right'  : (['bin_centers'], bins[1:])})
            return hist_ds
       
        # time splitting - hists are loaded for other variables
        #                - maybe move this to calcs section and save as file...
        dbdx = dbdx.swap_dims({'distance':'time'})

        # base list of weeks for rolling
        date_list = [np.datetime64('2018-12-10 00:00:00') +
                     np.timedelta64(i, 'W')
                     for i in range(16)]
        # mid week dates
        mid_date = [date_list[i] + (date_list[i+1] - date_list[i])/2
                   for i in range(15)]

        if by_time == 'weekly':
            # split into groups of weeks
            hist_ds = dbdx.resample(time='1W', skipna=True).map(
                                                         get_hist)
        elif by_time == '1W_rolling':
            # split into 1 week samples, sampled by week
            hist_ds = dbdx.groupby_bins('time', date_list,
                                    labels=mid_date).map(get_hist)
            hist_ds = hist_ds.rename({'time_bins':'time'})
        elif by_time == '2W_rolling':
            # split into 2 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::2] + np.timedelta64(84, 'h')
            l_label = mid_date[::2]
            hist_ds_l = dbdx.groupby_bins('time', l_dl,
                         labels=l_label).map(get_hist)
            u_dl = date_list[1:-1:2] + np.timedelta64(84, 'h')
            u_label = mid_date[1:-1:2]# + np.timedelta64(1, 'W')
            hist_ds_u = dbdx.groupby_bins('time', u_dl,
                         labels=u_label).map(get_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_l])
            hist_ds = hist_ds.rename({'time_bins':'time'})
        elif by_time == '3W_rolling':
            # split into 3 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::3]
            l_label = mid_date[::3]
            hist_ds_l = dbdx.groupby_bins('time', l_dl,
                         labels=l_label).map(get_hist)
            m_dl = date_list[1:-1:3]
            m_label = mid_date[1:-1:3]
            hist_ds_m = dbdx.groupby_bins('time', m_dl,
                         labels=m_label).map(get_hist)
            u_dl = date_list[2:-1:3]
            u_label = mid_date[2:-1:3]
            hist_ds_u = dbdx.groupby_bins('time', u_dl,
                         labels=u_label).map(get_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_m, hist_ds_l])
            hist_ds = hist_ds.rename({'time_bins':'time'})

        # plot over rolling intervals
        for i, (_,t) in enumerate(hist_ds.groupby('time')):
            if i == 15: continue
            ax = self.axs.flatten()[i]
            ax.vlines(t.hist, t.bin_left, t.bin_right,
                      transform=ax.transData, colors='orange', lw=0.8,
                      label='Giddy et al. (2020)')

        # entire timeseries
        hist_ds = get_hist(dbdx)
        ax = self.axs[-1,-1]
        ax.vlines(hist_ds.hist, hist_ds.bin_left, hist_ds.bin_right,
                  transform=ax.transData, colors='orange', lw=0.8,
                  label='Giddy et al. (2020)')

    def plot_histogram_buoyancy_gradients_and_samples_over_time(self, case,
                                       by_time):
        ''' 
        plot histogram of buoyancy gradients in week portions
        '''

        self.path = self.data_path + case
        self.figure, self.axs = plt.subplots(2,8, figsize=(6.5,3.5))
        plt.subplots_adjust(wspace=0.3, bottom=0.15, left=0.08, right=0.98,
                            top=0.95)
        #self.add_giddy(self.axs[0,0])
        self.add_giddy(by_time=by_time)

        sample_sizes = [1, 4, 20]
        #colours = ['g', 'b', 'r', 'y', 'c']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']

        for i, n in enumerate(sample_sizes):
            print ('sample', i)
            self.render_glider_sample_set_v(n=n, c=colours[i], style='bar',
                                            by_time=by_time)
        self.add_model_means_v(style='bar', by_time=by_time)
        self.add_giddy(by_time=by_time)

        for ax in self.axs[:,0]:
            ax.set_ylabel('Buoyancy Gradient')
        for ax in self.axs[1]:
            ax.set_xlabel('PDF', labelpad=12)

        #plt.legend()
        for ax in self.axs.flatten():
            ax.set_ylim(self.hist_range[0], self.hist_range[1])
            ax.set_xlim(0, 3e8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
        for ax in self.axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in self.axs[0]:
            ax.set_xticklabels([])
        
        if self.bg_method == 'norm':
            norm_str = '_pre_norm'
        else:
            norm_str = '_vector_mean'

        interp_str = self.interp

        plt.savefig(case + '_bg_sampling_skill' + self.append + '_'
                    + by_time + norm_str + interp_str + '.png', dpi=600)

    def plot_histogram_bg_pdf_averaged_weekly_samples(self, case, var='b_x_ml'):
        '''
        Plot buoyancy gradients over rolling frequency averaged over
        all rolling objects. Plots different ensemble sizes and different
        sampling lengths
        '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        self.figure, self.axs = plt.subplots(1,4, figsize=(6.5,3.0))
        plt.subplots_adjust(wspace=0.3, bottom=0.14, left=0.1, right=0.91,
                            top=0.95)
        #self.add_giddy(self.axs[0,0])
        #self.add_giddy(by_time=by_time)

        sample_sizes = [1, 4, 20]
        #colours = ['g', 'b', 'r', 'y', 'c']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']

        # entier glider time series
        # load glider data 
        self.ds_all = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_' + var + '.nc')
        self.ds_rolling = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_' + var + '.nc')

        # render
        self.render_glider_sample_set_averaged_over_weeks(self.axs,
                                                          sample_sizes, 
                                                          c=colours)
        self.add_model_means_averaged_over_weeks(self.axs, c='orange')
        #self.add_giddy(by_time=by_time)

        self.axs[0].set_ylabel(
                  r'$|\nabla b|$ [$\times 10^{-8}$ s$^{-1}$]')
        for ax in self.axs:
            ax.set_xlabel(r'PDF [$\times 10 ^{-8}$]', labelpad=12)

        # legend
        self.axs[-1].legend(loc='upper left', bbox_to_anchor=(0.80,1.0),
                            fontsize=6, borderaxespad=0)

        for ax in self.axs:
            ax.set_ylim(self.hist_range[0], self.hist_range[1])
            ax.set_xlim(0, 3e8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.xaxis.get_offset_text().set_visible(False)
        self.axs[0].yaxis.get_offset_text().set_visible(False)

        # align labels
        ypos = -0.1  # axes coords
        for ax in self.axs:
            ax.xaxis.set_label_coords(0.5, ypos)

        for ax in self.axs[1:]:
            ax.set_yticklabels([])

        # add time span info
        time_txt = ['1 week', '2 week', '3 week', '3.5 month']
        for i, txt in enumerate(time_txt):
            self.axs[i].text(0.3, 0.95, txt,
                          transform=self.axs[i].transAxes,
                          fontsize=6)

        if self.bg_method == 'norm':
            norm_str = '_pre_norm'
        else:
            norm_str = '_vector_mean'

        interp_str = self.interp

        plt.savefig(case + '_bg_sampling_skill_time_mean_' +
                    var + self.append + '_'
                    + norm_str + interp_str + '.png', dpi=600)

    def plot_histogram_bg_pdf_averaged_weekly_samples_multi_var(self, case):
        '''
        Plot buoyancy gradients over rolling frequency averaged over
        all rolling objects. Plots different ensemble sizes and different
        sampling lengths

        Use multiple variables
        '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        self.figure, self.axs = plt.subplots(2,4, figsize=(5.5,4.5))
        plt.subplots_adjust(wspace=0.1, bottom=0.08, left=0.13, right=0.98,
                            top=0.94, hspace=0.05)
        #self.add_giddy(self.axs[0,0])
        #self.add_giddy(by_time=by_time)

        sample_sizes = [1, 4, 20]
        #colours = ['g', 'b', 'r', 'y', 'c']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']

        def render(row, var):
            # load glider data 
            self.ds_all = xr.open_dataset(self.preamble + 'hist' 
                     + self.interp + self.append + '_full_time_' + var + '.nc')
            self.ds_rolling = xr.open_dataset(self.preamble + 'hist' 
                     + self.interp + self.append + '_rolling_' + var + '.nc')

            # render
            lines = self.render_glider_sample_set_averaged_over_weeks(row,
                                                              sample_sizes, 
                                                              c=colours)
            l = self.add_model_means_averaged_over_weeks(row, c='orange')

            return [l] + lines
            #self.add_giddy(by_time=by_time)

        # render variables
        p = render(self.axs[0], var='b_x_ml')
        render(self.axs[1], var='bg_norm_ml')

        l2 = r'$|\nabla b|$ ($\times 10^{-8}$ s$^{-2}$)'
        self.axs[0,0].set_ylabel('Along-Track Sampled\n' + l2)
        self.axs[1,0].set_ylabel('Across-Front Sampled\n' + l2)
        for ax in self.axs[1]:
            ax.set_xlabel(r'PDF ($\times 10 ^{-8}$)', labelpad=12)

        # legend
        #self.axs[0,-1].legend(loc='upper left', bbox_to_anchor=(1.02,1.0),
        #                    fontsize=6, borderaxespad=0)
        self.figure.legend(p, ['Model','1 Glider', '4 Gliders', '20 Gliders'],
                       loc='lower center', bbox_to_anchor=(0.555, 0.94), 
                       ncol=4, fontsize=8)

        for ax in self.axs.flatten():
            ax.set_ylim(self.hist_range[0], self.hist_range[1])
            ax.set_xlim(0, 2e8)
            #ax.spines['right'].set_visible(False)
            #ax.spines['top'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.xaxis.get_offset_text().set_visible(False)
        for row in self.axs:
            row[0].yaxis.get_offset_text().set_visible(False)
        for col in self.axs[0]:
            col.set_xticklabels([])

        # align labels
        ypos = -0.1  # axes coords
        for ax in self.axs[1]:
            ax.xaxis.set_label_coords(0.5, ypos)

        for ax in self.axs[:,1:].flatten():
            ax.set_yticklabels([])

        # add time span info
        time_txt = ['1-Week', '2-Week', '3-Week', '3.5-Month']
        letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
        for j, row in enumerate(self.axs):
            for i, txt in enumerate(time_txt):
                row[i].text(0.95, 0.98, 
                          letters[i + (4*j)] + '\n' + txt + '\nDeployment',
                          transform=row[i].transAxes,
                          fontsize=6, ha='right', va='top')
        #self.figure.text(0.5, 0.99, 'Along-Track Sampling', fontsize=8, 
        #                 ha='center', va='top', fontweight='bold')
        #self.figure.text(0.5, 0.50, 'Across-Front Sampling', fontsize=8, 
        #                 ha='center', fontweight='bold')

        if self.bg_method == 'norm':
            norm_str = '_pre_norm'
        else:
            norm_str = '_vector_mean'

        interp_str = self.interp

        plt.savefig(case + '_bg_sampling_skill_time_mean_multi_var_' +
                     self.append + norm_str + interp_str + '.png', dpi=600)

    def print_bg_rmse_averaged_weekly_samples_multi_var(self,case):
        ''' print rmse error for paper table '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        # get data
        b_x_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_b_x_ml.nc')
        bg_norm_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_bg_norm_ml.nc')
        b_x_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_b_x_ml.nc')
        bg_norm_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_bg_norm_ml.nc')

        # select rmse
        b_x_roll     = b_x_roll.rmse_mean
        bg_norm_roll = bg_norm_roll.rmse_mean
        b_x_full     = b_x_full.rmse_mean
        bg_norm_full = bg_norm_full.rmse_mean

        # restict to 1-5 gliders
        b_x_roll     = b_x_roll.sel(glider_quantity=slice(1,5))
        bg_norm_roll = bg_norm_roll.sel(glider_quantity=slice(1,5))
        b_x_full     = b_x_full.sel(glider_quantity=slice(1,5))
        bg_norm_full = bg_norm_full.sel(glider_quantity=slice(1,5))

        # mean
        b_x_f_mean = b_x_full.mean('bin_centers')
        bg_norm_f_mean = bg_norm_full.mean('bin_centers')
        b_x_r_mean = b_x_roll.mean(['bin_centers','time_counter'])
        bg_norm_r_mean = bg_norm_roll.mean(['bin_centers','time_counter'])
        print (b_x_f_mean.round(0))
        print (bg_norm_f_mean.round(0))
        print (b_x_r_mean.round(0))
        print (bg_norm_r_mean.round(0))
    
    def plot_histogram_bg_rmse_averaged_weekly_samples_multi_var(self, case):
        '''
        Plot percentage error of bg histograms ~~ glider versus model ~~.
   
        3 x 2 plot
        - Top row is line plots of percentage error against bg grad
          for different glider numbers, with diff_bg_norm and diff_b_x
        - Bottom row is mean across bg of the above percentage error
        Across rows changes samplign length from 1 week up to the 
        full 3.5 months.
        '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        ## initialise plot
        #fig, axs = plt.subplots(2,5, figsize=(6.5,4))
        #plt.subplots_adjust()

        # initialised figure
        fig = plt.figure(figsize=(5.5, 6.0))
        gs0 = gridspec.GridSpec(ncols=4, nrows=2)
        gs1 = gridspec.GridSpec(ncols=2, nrows=1)
        gs0.update(top=0.95, bottom=0.57, left=0.12, right=0.98, hspace=0.1,
                   wspace=0.1)
        gs1.update(top=0.39, bottom=0.08, left=0.12, right=0.98, wspace=0.1)

        axs0, axs1 = [], []
        for i in range(8):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(2):
            axs1.append(fig.add_subplot(gs1[i]))

        # get data
        b_x_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_b_x_ml.nc')
        bg_norm_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_bg_norm_ml.nc')
        b_x_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_b_x_ml.nc')
        bg_norm_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_bg_norm_ml.nc')
        

        # plot rmse across bg
        nums = [1,4,20]
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        p = []
        for i, num in enumerate(nums):
            b_x = b_x_full.sel(glider_quantity=num)
            print ('')
            print ('')
            print ('')
            print ('')
            print (b_x.rmse_mean.max().values)
            l = axs0[3].bar(b_x.bin_left, 
                        b_x.rmse_mean, 
                        color=colours[i],
                        width=b_x.bin_right - b_x.bin_left,
                        align='edge')

            bg_norm = bg_norm_full.sel(glider_quantity=num)
            axs0[7].bar(bg_norm.bin_left, 
                        bg_norm.rmse_mean, 
                        color=colours[i],
                        width=bg_norm.bin_right - bg_norm.bin_left,
                        align='edge')
                        #label='gliders: ' + str(n))
            p.append(l)

            for j, roll in enumerate(['1W_rolling','2W_rolling','3W_rolling']):
                b_x = b_x_roll.sel(glider_quantity=num, rolling=roll)
                b_x = b_x.mean('time_counter')
                print (b_x.rmse_mean.max().values)
                print ('')
                axs0[j].bar(b_x.bin_left, 
                             b_x.rmse_mean, 
                             color=colours[i],
                             width=b_x.bin_right - b_x.bin_left,
                             align='edge')

                bg_norm = bg_norm_roll.sel(glider_quantity=num, rolling=roll)
                bg_norm = bg_norm.mean('time_counter')
                axs0[j+4].bar(bg_norm.bin_left, 
                             bg_norm.rmse_mean, 
                             color=colours[i],
                             width=bg_norm.bin_right - bg_norm.bin_left,
                             align='edge')

        print ('')
        print ('')
        print ('')
        print ('')
        fig.legend(p, ['1 Glider', '4 Gliders', '20 Gliders'],
                       loc='lower center', bbox_to_anchor=(0.555, 0.95), 
                       ncol=4, fontsize=8)

        # ~~~ plot rmse across glider number ~~~ #

        c1 = '#f18b00'
        colours=[c1, 'purple', 'green' 'navy']#,'turquoise']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        c0 = '#7e9aa5'
        #c0 = '#dad1d1'
        c= ['lightgrey', 'grey', 'black', c1]
        #c = ['#dad1d1', '#7e9aa5', '#55475a', c1]
        #c=[c1, 'purple', 'green', 'navy']#,'turquoise']
        #cg_0='grey'
        #cg_1='black'
        #cg_2='lightgrey'

        b_x_f_mean = b_x_full.mean('bin_centers')
        bg_norm_f_mean = bg_norm_full.mean('bin_centers')
        b_x_r_mean = b_x_roll.mean(['bin_centers','time_counter'])
        bg_norm_r_mean = bg_norm_roll.mean(['bin_centers','time_counter'])

        # cut small bg magnitudes
        b_x_full_cut = b_x_full.isel(bin_centers=slice(3,None))
        b_x_roll_cut = b_x_roll.isel(bin_centers=slice(3,None))
        b_x_f_mean_cut = b_x_full_cut.mean('bin_centers')
        b_x_r_mean_cut = b_x_roll_cut.mean(['bin_centers','time_counter'])

        bg_norm_full_cut = bg_norm_full.isel(bin_centers=slice(3,None))
        bg_norm_roll_cut = bg_norm_roll.isel(bin_centers=slice(3,None))
        bg_norm_f_mean_cut = bg_norm_full_cut.mean('bin_centers')
        bg_norm_r_mean_cut=bg_norm_roll_cut.mean(['bin_centers','time_counter'])

        p0, = axs1[0].plot(b_x_f_mean.glider_quantity, b_x_f_mean.rmse_mean,
                     c=c[3], lw=1.5, zorder=10)
        p1, = axs1[0].plot(b_x_f_mean_cut.glider_quantity,
                     b_x_f_mean_cut.rmse_mean,
                     c=c[3], ls='--', lw=1.5)
        axs1[1].plot(bg_norm_f_mean.glider_quantity, bg_norm_f_mean.rmse_mean,
                     c=c[3], lw=1.5)

        p = []
        for i, roll in enumerate(['1W_rolling','2W_rolling','3W_rolling']):
            b_x_mean = b_x_r_mean.sel(rolling=roll)
            b_x_mean_cut = b_x_r_mean_cut.sel(rolling=roll)
            bg_norm_mean = bg_norm_r_mean.sel(rolling=roll)
            bg_norm_mean_cut = bg_norm_r_mean_cut.sel(rolling=roll)
            l, = axs1[0].plot(b_x_mean.glider_quantity, b_x_mean.rmse_mean, 
                         c=c[i], lw=1.5)
            axs1[1].plot(bg_norm_mean.glider_quantity, bg_norm_mean.rmse_mean,
                         c=c[i], lw=1.5)
            p.append(l)
            #ax.text(0.9,0.9,roll,transform=ax.transAxes)

        p = p + [p0]# + [p1]
        #labs = ['1-Week', '2-Week', '3-Week', '3.5-Month', '3.5-Month '+ thresh]
        labs = ['1-Week', '2-Week', '3-Week', '3.5-Month', '3.5-Month']
        fig.legend(p, labs, loc='lower center', title='Deployment',
                       title_fontsize=8,
                       bbox_to_anchor=(0.555, 0.39), ncol=4, fontsize=8)

        for ax in axs0:
            ax.set_ylim(0,160)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.set_xlim(self.hist_range[0], self.hist_range[1])
        for ax in axs0[:4]:
            ax.set_xticklabels([])
        for ax in axs0[1:4] + axs0[5:] + [axs1[1]]:
            ax.set_yticklabels([])
        for ax in axs1:
            ax.set_ylim(0,120)

        for ax in axs1:
            ax.set_xlim(1,30)
            ax.set_xlabel('Number of Gliders')
            ax.set_xticks([1,5,10,15,20,25,30])

        l0 = r'$|\nabla b|$'
        l1 = r'($\times 10^{-8}$ s$^{-2}$)'
        axs0[4].set_xlabel(l0 + ' 1-Week\n' + l1)
        axs0[5].set_xlabel(l0 + ' 2-Week\n' + l1)
        axs0[6].set_xlabel(l0 + ' 3-Week\n' + l1)
        axs0[7].set_xlabel(l0 + ' 3.5-Month\n' + l1)

        l2 = ' RMSE (%)'
        axs0[0].set_ylabel('Along-Track\nSampled' + l2)
        axs0[4].set_ylabel('Across-Front\nSampled'+ l2)

        l2 = ' RMSE (%)'
        axs0[0].set_ylabel('Along-Track\nSampled' + l2)
        axs0[4].set_ylabel('Across-Front\nSampled'+ l2)
       
        axs1[0].set_ylabel('Mean RMSE (%)')

        axs1[0].text(0.98, 0.96, 'Along-Track', va='top', ha='right',
                    transform=axs1[0].transAxes)
        axs1[1].text(0.98, 0.96, 'Across-Front', va='top', ha='right',
                    transform=axs1[1].transAxes)

        thresh = r'$|\nabla b|> 4 \times 10^{-9}$ s$^{-2}$'
        axs1[0].text(29.5, 32, thresh, va='bottom', ha='right',
                    transform=axs1[0].transData, c=c[3], fontsize=6)
        # letters
        letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        for i, ax in enumerate(axs0):
            ax.text(0.95, 0.95, letters[i], transform=ax.transAxes, va='top',
                    ha='right')

        letters = ['(i)', '(j)']
        for i, ax in enumerate(axs1):
            ax.text(0.03, 0.96, letters[i], transform=ax.transAxes, va='top',
                    ha='left')

        plt.savefig('paper_hist_rmse.png', dpi=1200)

    def plot_parallel_path_rmse(self, case):
        '''
        Plot RMSE error as a percentage against the model mean for parallel
        paths.

        This is a 2x1 plot. Panel 1 is along-track buoyancy gradients. Panel 2
        is across transect buoyancy gradients.
        '''

        # initialise figure
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5.5,3.0))
        plt.subplots_adjust(wspace=0.1, bottom=0.15, left=0.10, right=0.98,
                            top=0.85)

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case + '/BGHists' 
        self.preamble = self.path + file_id

        # get data
        b_x_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_b_x_ml.nc')
        b_x_ct_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_b_x_ct_12_ml.nc')
        b_x_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_b_x_ml.nc')
        b_x_ct_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_b_x_ct_12_ml.nc')

        # set colours
        c1 = '#f18b00'
        colours=[c1, 'purple', 'green' 'navy']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        c0 = '#7e9aa5'
        c= ['lightgrey', 'grey', 'black', c1]

        # get means
        b_x_f_mean = b_x_full.mean('bin_centers')
        b_x_ct_f_mean = b_x_ct_full.mean('bin_centers')
        b_x_r_mean = b_x_roll.mean(['bin_centers','time_counter'])
        b_x_ct_r_mean = b_x_ct_roll.mean(['bin_centers','time_counter'])

        # plot full time mean
        p0, = ax0.plot(b_x_f_mean.glider_quantity, b_x_f_mean.rmse_mean,
                     c=c[3], lw=1.5, zorder=10)
        ax1.plot(b_x_ct_f_mean.glider_quantity, b_x_ct_f_mean.rmse_mean,
                     c=c[3], lw=1.5)

        # plot rolling time means
        p = []
        for i, roll in enumerate(['1W_rolling','2W_rolling','3W_rolling']):
            b_x_mean = b_x_r_mean.sel(rolling=roll)
            b_x_ct_mean = b_x_ct_r_mean.sel(rolling=roll)
            l, = ax0.plot(b_x_mean.glider_quantity, b_x_mean.rmse_mean, 
                         c=c[i], lw=1.5)
            ax1.plot(b_x_ct_mean.glider_quantity, b_x_ct_mean.rmse_mean,
                         c=c[i], lw=1.5)
            p.append(l)

        p = p + [p0]
        labs = ['1-Week', '2-Week', '3-Week', '3.5-Month', '3.5-Month']
        fig.legend(p, labs, loc='lower center', title='Deployment',
                       title_fontsize=8,
                       bbox_to_anchor=(0.555, 0.85), ncol=4, fontsize=8)

        # axes settings
        for ax in [ax0, ax1]:
            ax.set_xlim(1,30)
            ax.set_xlabel('Number of Gliders')
            ax.set_xticks([1,5,10,15,20,25,30])
            ax.set_ylim(0,120)
        ax1.set_yticklabels([])

        l0 = r'$|\nabla b|$'
        l1 = r'($\times 10^{-8}$ s$^{-2}$)'

        ax0.set_ylabel('Mean RMSE (%)')

        ax0.text(0.98, 0.96, 'Along-Track', va='top', ha='right',
                    transform=ax0.transAxes)
        ax1.text(0.98, 0.96, 'Across-Deployment', va='top', ha='right',
                    transform=ax1.transAxes)

        # letters
        letters = ['(a)', '(b)']
        for i, ax in enumerate([ax0,ax1]):
            ax.text(0.03, 0.96, letters[i], transform=ax.transAxes, va='top',
                    ha='left')

        plt.savefig('paper_hist_parallel_transects_rmse.png', dpi=1200)

    def get_ensembles(self, case, by_time):
        '''
        load weekly model hists and calculate mean buoyancy gradients in x and y
            - for rmse differences
        '''

        # load weekly model hists
        m = xr.open_dataset(self.data_path + case + 
                     '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                     self.append + '_' + by_time + '.nc')#.isel(
                    # time_counter=slice(None,-1))
        
        # pre merge function
        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds
         
        # load weekly glider hists
        prep = case + '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        ensemble_list = [self.data_path + prep + str(i).zfill(2) + '_hist' + 
                         self.append + '_' + by_time + '.nc'
                         for i in range(1,31)]
        ensembles = xr.open_mfdataset(ensemble_list, 
                                   combine='nested', concat_dim='ensemble_size',
                                     preprocess=pre_proc).load()
        ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))
        ensembles = ensembles.set_coords('sample_size')

        # mean of the histograms
        if self.bg_method == 'mean':
            ensembles['m_bg_abs'] = (m.hist_x + m.hist_y) / 2

        # norm of the vectors - calculated before hist
        elif self.bg_method == 'norm':
            ensembles['m_bg_abs'] = m.hist_norm

        return ensembles

    def rmsep(self, pred, truth):
        ''' calulate root mean squared percentage error '''

        norm = (pred - truth)/truth 

        return np.sqrt(((norm)**2).mean(dim='bin_centers')) * 100

    def get_spatial_mean_and_std_bg(self, case):
        '''
        calcualte spatial mean and standard deviation of buoyancy gradients
        at 10 m depth
        '''

        # load bg
        bg = xr.open_dataset(self.data_path + case +
                             self.file_id + 'bg_z10m.nc', chunks='auto')
        # cut edges of domain
        bg = bg.isel(x=slice(20,-20), y=slice(20,-20))
  
        # absolute value of vector mean
        if self.bg_method == 'mean':
            bg = np.abs((bg.bx + bg.by)/2).load()

        # bg normed
        elif self.bg_method == 'norm':
            bg = ((bg.bx**2 + bg.by**2) ** 0.5).load()

        else:
            print ('method not recognised')

        bg_mean = bg.mean(['x','y'])
        bg_std = bg.std(['x','y'])
 
        return bg_mean, bg_std

    def plot_rmse_over_ensemble_sizes_and_week(self, case, by_time):
        ''' plot the root mean squared error of the 1 s.d. (? not decile)
            from the **real** mean over week and ensemble size
            contourf
        '''

        ensembles = self.get_ensembles(case, by_time, method='norm')

        # calculate rmse
        rmse_l = self.rmsep(ensembles.hist_l_dec, ensembles.m_bg_abs)
        rmse_u = self.rmsep(ensembles.hist_u_dec, ensembles.m_bg_abs)
        rmse_mean = self.rmsep(ensembles.hist_mean, ensembles.m_bg_abs)

        # initialised figure
        fig = plt.figure(figsize=(6.5, 4), dpi=300)
        gs0 = gridspec.GridSpec(ncols=1, nrows=2)
        gs1 = gridspec.GridSpec(ncols=1, nrows=1)
        gs0.update(top=0.98, bottom=0.35, left=0.13, right=0.87, hspace=0.1)
        gs1.update(top=0.30, bottom=0.15, left=0.13, right=0.87)

        axs0 = []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        axs1 = fig.add_subplot(gs1[0])

        # initialise plot
        #fig, axs = plt.subplots(2, figsize=(6.5,4))
        #plt.subplots_adjust(left=0.08, right=0.87, hspace=0.1, bottom=0.15,
        #                    top=0.98)

        # render
        cmap = plt.cm.inferno
        lev = np.linspace(0,300,11)
        p0 = axs0[0].contourf(rmse_u.time_counter, rmse_u.ensemble_size, rmse_u,
                             levels=lev, cmap=cmap)
        lev = np.linspace(0,100,11)
        p1 = axs0[1].contourf(rmse_l.time_counter, rmse_l.ensemble_size, rmse_l,
                             levels=lev, cmap=cmap)

        # colour bar upper
        pos = axs0[0].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.1, 0.5, 'RMSE of\nbuoyancy gradients (%)', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # colour bar lower
        pos = axs0[1].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.1, 0.5, 'RMSE of\nbuoyancy gradients (%)', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # text labels
        axs0[0].text(0.99, 0.98, 'upper decile', c='w', va='top', ha='right',
                    transform=axs0[0].transAxes)
        axs0[1].text(0.99, 0.98, 'lower decile', c='w', va='top', ha='right',
                    transform=axs0[1].transAxes)

        # axes labels
        for ax in axs0:
            ax.set_ylabel('ensemble size')
            ax.set_xticks(rmse_mean.time_counter)

        # set xlabels
        axs0[0].set_xticklabels([])
        axs0[1].set_xticklabels([])


        # add time series of sample size
        sample_size = ensembles.sample_size.isel(ensemble_size=0)
        axs1.plot(sample_size.time_counter, sample_size)

        # add time series of bg - standard deviation
        _, std = self.get_spatial_mean_and_std_bg(case)
        axs1_2 = axs1.twinx()
        axs1_2.plot(std.time_counter, std)

        # set xticks
        week_labels = sample_size.time_counter.dt.strftime('%m-%d').values
        axs1.set_xlim(sample_size.time_counter.min(),
                      sample_size.time_counter.max())
        axs1.set_ylim(0,500)
        axs1.set_xticklabels(week_labels)
        axs1.set_xticks(sample_size.time_counter)
        axs1.set_xlabel('date (MM-DD)')
        axs1.set_ylabel('sample\nsize')

        # rotate labels
        for label in axs1.get_xticklabels(which='major'):
            label.set(rotation=35, horizontalalignment='right')

        plt.savefig(case + '_bg_RMSE_' + by_time 
                    + self.append + '.png', dpi=600)

    def plot_rmse_over_ensemble_sizes_and_week_3_panel(self, case):
        ''' plot the root mean squared error of the 1 s.d. (? not decile)
            from the **real** mean over week and ensemble size
            contourf
        '''

        # initialised figure
        fig = plt.figure(figsize=(6.5, 4), dpi=300)
        gs0 = gridspec.GridSpec(ncols=3, nrows=2)
        gs1 = gridspec.GridSpec(ncols=3, nrows=1)
        gs0.update(top=0.95, bottom=0.35, left=0.08, right=0.86, hspace=0.1,
                   wspace=0.07)
        gs1.update(top=0.32, bottom=0.15, left=0.08, right=0.86,
                   wspace=0.07)

        axs0, axs1 = [], []
        for j in range(3):
            for i in range(2):
                axs0.append(fig.add_subplot(gs0[i,j]))
        for i in range(3):
            axs1.append(fig.add_subplot(gs1[i]))

        # render
        def calc_and_render(by_time, a0, a1):

            # load data
            ensembles = self.get_ensembles(case, by_time, method='norm')

            # calculate rmse
            rmse_l = self.rmsep(ensembles.hist_l_dec, ensembles.m_bg_abs)
            rmse_u = self.rmsep(ensembles.hist_u_dec, ensembles.m_bg_abs)
            rmse_mean = self.rmsep(ensembles.hist_mean, ensembles.m_bg_abs)

            cmap = plt.cm.inferno

            # upper decile
            lev = np.linspace(0,300,11)
            p0 = a0[0].contourf(rmse_u.time_counter, rmse_u.ensemble_size,
                                  rmse_u, levels=lev, cmap=cmap)

            # lower decile
            lev = np.linspace(0,100,11)
            p1 = a0[1].contourf(rmse_l.time_counter, rmse_l.ensemble_size,
                                    rmse_l, levels=lev, cmap=cmap)

            # add time series of sample size
            sample_size = ensembles.sample_size.isel(ensemble_size=0)
            p2, = a1.plot(sample_size.time_counter, sample_size, lw=0.8)
            a1.yaxis.label.set_color(p2.get_color())
            a1.tick_params(axis='y', colors=p2.get_color())

            return p0, p1, ensembles.time_counter

        # render each column
        calc_and_render('1W_rolling', axs0[:2], axs1[0])
        calc_and_render('2W_rolling', axs0[2:4], axs1[1])
        p0, p1, time_counter = calc_and_render('3W_rolling', axs0[4:], axs1[2])

        # add time series of bg - standard deviation
        _, std = self.get_spatial_mean_and_std_bg(case)
        twin_axes, p  = [], []
        for ax in axs1[:2]:
            a1_2 = ax.twinx()
            twin_axes.append(a1_2)
            p.append(a1_2.plot(std.time_counter, std, c='g', lw=0.8)[0])
            a1_2.set_yticklabels([])
        a1_2 = axs1[-1].twinx()
        twin_axes.append(a1_2)
        p.append(a1_2.plot(std.time_counter, std, c='g', lw=0.8)[0])
        a1_2.set_ylabel(r'$\sigma_{bg}$' + '\n' + r'$[\times 10^{-8}]$')
        a1_2.yaxis.get_offset_text().set_visible(False)
        for i, ax in enumerate(twin_axes):
            ax.yaxis.label.set_color(p[i].get_color())
            ax.tick_params(axis='y', colors=p[i].get_color())

        # top two rows
        for ax in axs0:
            ax.set_xticklabels([])
            ax.set_xticks(time_counter)
        for ax in axs0[:2]:
            ax.set_ylabel('ensemble size')
        for ax in axs0[2:]:
            ax.set_yticklabels([])

        # bottom row axes details
        for ax in axs1:
            week_labels = time_counter.dt.strftime('%m-%d').values
            week_labels[::2] = ''
            ax.set_xlim(time_counter.min(), time_counter.max())
            ax.set_ylim(0,500)
            ax.set_xticklabels(week_labels)
            ax.set_xticks(time_counter)
            ax.set_xlabel('date (MM-DD)')

            # rotate labels
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=35, horizontalalignment='right')

        axs1[0].set_ylabel('sample size')
        for ax in axs1[1:]:
            ax.set_yticklabels([])

        # colour bar upper
        pos = axs0[4].get_position()
        cbar_ax = fig.add_axes([0.87, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(3.7, 0.5, 'RMSE of\nbuoyancy gradients\n[%]', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # colour bar lower
        pos = axs0[5].get_position()
        cbar_ax = fig.add_axes([0.87, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(3.7, 0.5, 'RMSE of\nbuoyancy gradients\n[%]', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # text labels
        by_time_labels = ['1-Week Rolling', '2-Week Rolling', '3-Week Rolling']
        for i, ax in enumerate(axs0[::2]):
            ax.text(0.99, 0.98, 'upper decile', c='w', va='top', ha='right',
                    transform=ax.transAxes, fontsize=6)
            ax.text(0.5, 1.01, by_time_labels[i], va='bottom', ha='center',
                    transform=ax.transAxes)
        for ax in axs0[1::2]:
            ax.text(0.99, 0.98, 'lower decile', c='w', va='top', ha='right',
                    transform=ax.transAxes, fontsize=6)

        # align labels
        xpos = -0.21  # axes coords
        for ax in axs0[:2]:
            ax.yaxis.set_label_coords(xpos, 0.5)
        axs1[0].yaxis.set_label_coords(xpos, 0.5)

        plt.savefig(case + '_bg_RMSE_3_panel_' + self.append + 'pre_norm.png',
                    dpi=600)


    def plot_correlation_rmse(self, case):
        '''
        scatter plots of RMSE against
            - ensemble size
            - sample size
            - spatial? standard deviation in bg
        '''
 
        # initialise figure
        fig, axs = plt.subplots(3, 1, figsize=(3.2,3.5))
        plt.subplots_adjust(top=0.95, bottom=0.13, left=0.09, right=0.78,
                            hspace=0.15)

        # get standard deviation of bg for correlations
        _, std = self.get_spatial_mean_and_std_bg(case)

        def all_by_times(by_time, std, c='r', pos=1):

            ensembles = self.get_ensembles(case, by_time)
            std = std.interp(time_counter=ensembles.time_counter)

            # calculate rmse
            rmse_l = self.rmsep(ensembles.hist_l_dec, ensembles.m_bg_abs)
            rmse_u = self.rmsep(ensembles.hist_u_dec, ensembles.m_bg_abs)

            # drop 100 % errors
            rmse_l = rmse_l.where(rmse_l != 100)
            rmse_u = rmse_u.where(rmse_u != 100)
            rmse_l['bg_std'] = std.interp(time_counter=rmse_l.time_counter)
            rmse_u['bg_std'] = std.interp(time_counter=rmse_u.time_counter)

            rmse_l = rmse_l.stack(z=['ensemble_size','time_counter'])
            rmse_u = rmse_u.stack(z=['ensemble_size','time_counter'])
            l_ensemble_size_norm=rmse_l.ensemble_size/rmse_l.ensemble_size.max()
            u_ensemble_size_norm=rmse_u.ensemble_size/rmse_u.ensemble_size.max()
            #sample_size_max = 448 # max for all rolling lengths
            #l_sample_size_norm = rmse_l.sample_size/sample_size_max
            #u_sample_size_norm = rmse_u.sample_size/sample_size_max
            l_sample_size_norm = rmse_l.sample_size/rmse_l.sample_size.max()
            u_sample_size_norm = rmse_u.sample_size/rmse_u.sample_size.max()
            l_bg_std_norm = rmse_l.bg_std/rmse_l.bg_std.max()
            u_bg_std_norm = rmse_u.bg_std/rmse_u.bg_std.max()
            normalise = matplotlib.colors.Normalize(vmin=0, vmax=448)

            m='|'
            s=5
            a=0.4
            axs[0].scatter(rmse_u, pos * np.ones(len(rmse_u)) + 0.2, 
                           c=rmse_u.ensemble_size, s=s, alpha=a, marker=m)
            axs[1].scatter(rmse_u, pos * np.ones(len(rmse_u)) + 0.2, 
                           c=rmse_u.sample_size, s=s, alpha=a, marker=m,
                           norm=normalise)
            axs[2].scatter(rmse_u, pos * np.ones(len(rmse_u)) + 0.2, 
                           c=rmse_u.bg_std, s=s, alpha=a, marker=m)
            p0 = axs[0].scatter(rmse_l, pos * np.ones(len(rmse_l)) - 0.2, 
                           c=rmse_l.ensemble_size, s=s, alpha=a, marker=m)
            p1 = axs[1].scatter(rmse_l, pos * np.ones(len(rmse_l)) - 0.2, 
                           c=rmse_l.sample_size, s=s, alpha=a, marker=m,
                           norm=normalise)
            p2 = axs[2].scatter(rmse_l, pos * np.ones(len(rmse_l)) - 0.2, 
                           c=rmse_l.bg_std, s=s, alpha=a, marker=m)

            # colour bar
            pos0 = axs[0].get_position()
            cbar_ax = fig.add_axes([0.86, pos0.y0, 0.02, pos0.y1 - pos0.y0])
            cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
            cbar.ax.text(5.3, 0.5, 'ensemble size',
                         fontsize=6, rotation=90, transform=cbar.ax.transAxes,
                         va='center', ha='left', multialignment='center')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            pos1 = axs[1].get_position()
            cbar_ax = fig.add_axes([0.86, pos1.y0, 0.02, pos1.y1 - pos1.y0])
            cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
            cbar.ax.text(5.3, 0.5, 'sample size',
                         fontsize=6, rotation=90, transform=cbar.ax.transAxes,
                         va='center', ha='left', multialignment='center')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            pos2 = axs[2].get_position()
            cbar_ax = fig.add_axes([0.86, pos2.y0, 0.02, pos2.y1 - pos2.y0])
            cbar = fig.colorbar(p2, cax=cbar_ax, orientation='vertical')
            cbar.ax.text(5.3, 0.5, r'$\sigma_{bg} [\times 10^{-8}]$',
                         fontsize=6, rotation=90, transform=cbar.ax.transAxes,
                         va='center', ha='left', multialignment='center')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.get_offset_text().set_visible(False)

            for ax in axs:
                ax.text(0, pos + 0.2, 'u', va='center', ha='center',
                        fontsize=6, transform=ax.transData)
                ax.text(0, pos - 0.2, 'l', va='center', ha='center',
                        fontsize=6, transform=ax.transData)
                ax.text(350, 4, r'$r$',
                        va='center', ha='center',
                        fontsize=6, transform=ax.transData)

            # pearsons rank correlation
            print (rmse_l)
            rmse_l = rmse_l.dropna('z')
            u_ens_pr = round(stats.pearsonr(rmse_u.ensemble_size, rmse_u)[0], 3)
            l_ens_pr = round(stats.pearsonr(rmse_l.ensemble_size, rmse_l)[0], 3)
            u_sam_pr = round(stats.pearsonr(rmse_u.sample_size, rmse_u)[0], 3)
            l_sam_pr = round(stats.pearsonr(rmse_l.sample_size, rmse_l)[0], 3)
            u_std_pr = round(stats.pearsonr(rmse_u.bg_std, rmse_u)[0], 3)
            l_std_pr = round(stats.pearsonr(rmse_l.bg_std, rmse_l)[0], 3)

            axs[0].text(350, pos + 0.2, str(u_ens_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[0].transData)
            axs[0].text(350, pos - 0.2, str(l_ens_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[0].transData)
            axs[1].text(350, pos + 0.2, str(u_sam_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[1].transData)
            axs[1].text(350, pos - 0.2, str(l_sam_pr),
                     va='center', ha='center',
                    fontsize=6, transform=axs[1].transData)
            axs[2].text(350, pos + 0.2, str(u_std_pr),
                     va='center', ha='center',
                     fontsize=6, transform=axs[2].transData)
            axs[2].text(350, pos - 0.2, str(l_std_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[2].transData)

        all_by_times('1W_rolling', std, c='r', pos=3)
        all_by_times('2W_rolling', std, c='g', pos=2)
        all_by_times('3W_rolling', std, c='b', pos=1)
 
        # axis details
        ax_labels = ['ensemble size', 'sample size', r'$\sigma_{bg}$']
        for i, ax in enumerate(axs):
            ax.set_xlim(0,335)
            ax.set_ylim(0,4)
            ax.set_yticks([1,2,3])
            ax.set_yticklabels(['3W', '2W', '1W'])
            ax.text(0.5, 1.01, ax_labels[i], va='bottom', ha='center',
                    fontsize=8, transform=ax.transAxes)

            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.tick_params(tick1On=False)
            #ax.tick_params(axis='y', which='both', length=0)
            ax.tick_params(left=False)

        axs[2].set_xlabel('RMSE of buoyancy gradients [%]')
        for ax in axs[:2]:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])

        plt.savefig('EXP10_bg_rmse_corr_pre_norm.png', dpi=600)
  
def plot_correlations():
    boot = bootstrap_plotting(bg_method='norm')
    boot.plot_correlation_rmse('EXP10')

#plot_correlations()

def plot_hist(by_time=None):
    cases = ['EXP10', 'EXP08', 'EXP13']
    cases = ['EXP10']
    if by_time:
        boot = bootstrap_plotting(bg_method='norm', interp='1000')
        #boot.plot_histogram_buoyancy_gradients_and_samples_over_time(
        #                                                      'EXP10', by_time)
        #boot.plot_rmse_over_ensemble_sizes_and_week('EXP10', by_time)
        #boot.plot_rmse_over_ensemble_sizes_and_week_3_panel('EXP10')
        #boot.plot_histogram_bg_pdf_averaged_weekly_samples('EXP10',
        #                                                          var='b_x_ml')
    
    else:
        boot = bootstrap_plotting(append='parallel_transects')
        #boot.plot_histogram_bg_pdf_averaged_weekly_samples_multi_var('EXP10')
        #boot.print_bg_rmse_averaged_weekly_samples_multi_var('EXP10')
        boot.plot_parallel_path_rmse('EXP10')
        #boot.plot_histogram_bg_rmse_averaged_weekly_samples_multi_var('EXP10')
            #m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=False,
            #                             subset='')
            #m.plot_histogram_buoyancy_gradients_and_samples()
            #m.plot_rmse_over_ensemble_sizes()

plot_hist()
