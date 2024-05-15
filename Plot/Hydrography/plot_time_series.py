import xarray as xr
import matplotlib.pyplot as plt
import config 
import matplotlib.dates as mdates
import numpy as np
import calendar as cal

class time_series(object):

    def __init__(self, cases, title_list):
        self.path = config.data_path()
        self.root = config.root()

        self.cases = {}

        for i, case in enumerate(cases):
            print (self.path + case)
            mean = xr.open_mfdataset(self.path + case + 
                               '/Stats/SOCHIC_PATCH_mean*.nc',
                               compat='override', coords='different')
                               #chunks={'time_counter':10})
            std = xr.open_mfdataset(self.path + case + 
                               '/Stats/SOCHIC_PATCH_std*.nc',
                               compat='override', coords='different',
                               chunks={'time_counter':10})
            self.cases[case] = xr.merge([mean, std])
            self.cases[case].attrs['title'] = title_list[i]

        #self.cases[year] = self.cases[year].drop_vars('time_instant')
        #self.cases[year] = xr.decode_cf(self.cases[year])
        #self.ds = self.cases[case]
        # buoyancy gradients
        #    if bg:
        #        bg_stats = xr.open_dataset(self.path + case + 
        #                                 '/buoyancy_gradient_stats.nc')
        #        self.cases[case] = xr.merge([bg_stats, self.cases[case]])

    def area_mean(self, var):
        ''' mean quantity over depth '''
   
        print (var)
        print (self.ds[var].values)
        #self.ds[var + '_mean'] = self.ds[var].mean(['x','y'])
   
        return self.ds[var].mean(['x','y']).load()

    def area_std(self, var):
        ''' mean quantity over depth '''
   
        #self.ds[var + '_std'] = self.ds[var].std(['x','y'])
        return self.ds[var].std(['x','y'], skipna=True).load()
   
    
    def render_panel(self, ax):
        ''' plot the area-mean mixed layer depth '''

        ax.plot(self.ds.time_counter, self.ds[var + '_mean'])

    def plot_heat_fluxes(self, year):
        ''' plot heat fluxes into ocean'''

        self.fig, self.axs = plt.subplots(3, 1, figsize=(5.5,4.0))
        plt.subplots_adjust(left=0.16)

        colours = ['royalblue', 'orange', 'red']
        text_pos = [0.05, 0.2, 0.35]

        for (case, data) in self.cases.items():
            for i, (year, ds) in enumerate(data.groupby('time_counter.year')): 
                colour = colours[i]
                self.axs[0].text(text_pos[i], 1.05, year, c=colour,
                            transform=self.axs[0].transAxes)

                qt_oce = ds.qt_oce_mean.rolling(time_counter=7,
                                                center=True).mean()
                self.axs[0].plot(qt_oce.time_counter.dt.dayofyear, qt_oce,
                                 lw=1, c=colour)
                qns_oce = ds.qns_oce_mean.rolling(time_counter=7,
                                                  center=True).mean()
                self.axs[1].plot(qns_oce.time_counter.dt.dayofyear, qns_oce,
                                 lw=1, c=colour)
                qsr_oce = ds.qsr_oce_mean.rolling(time_counter=7,
                                                  center=True).mean()
                self.axs[2].plot(qsr_oce.time_counter.dt.dayofyear, qsr_oce,
                                 lw=1, c=colour)
        for ax in self.axs:
        #    #ax.set_xlim([self.ds.time_counter.min(),
        #    #            self.ds.time_counter.max()])
            ax.set_xlim(0,366)
            ax.axhline(0, lw=0.5, c='black', ls=':', zorder=1)
            #major_format = mdates.DateFormatter('%b')
            #ax.xaxis.set_major_formatter(major_format)
        for ax in self.axs[:-1]:
            ax.set_xticks([])

        self.axs[0].set_ylabel('surface\n heat flux')
        self.axs[1].set_ylabel('non-solar\n heat flux')
        self.axs[2].set_ylabel('solar\n heat flux')

        self.axs[2].set_xlabel('day of year')

        plt.savefig('heat_fluxes_EXP04.png', dpi=600)

    def plot_mld_sip(self, years, giddy=False, orca=False, satellite=False,
                                  argo=False):

        self.fig, self.axs = plt.subplots(2,1, figsize=(5.5,4.0))

        colours = ['royalblue', 'orange', 'olivedrab']
        text_pos = [0.05, 0.18, 0.5]
        
        year_len = len(years)
         
        for j, (case, data) in enumerate(self.cases.items()):
            if year_len == 1:
                if case == 'EXP04':
                    data = data.sel(time_counter='2013')
                else:
                    data = data.sel(time_counter=years[0])
            # add case label
            if len(self.cases) > 1:
                colour = colours[j]
                self.axs[0].text(text_pos[j], 1.05, data.title,
                                 c=colour, transform=self.axs[0].transAxes)
            for i, (year, ds) in enumerate(data.groupby('time_counter.year')): 
                print ('year', year)
                # add year label
                if year_len > 1:
                    colour = colours[i]
                    self.axs[0].text(text_pos[i], 1.05, year, c=colour,
                                transform=self.axs[0].transAxes)
                print ('mld_std')
                upper = (ds.mldr10_3_mean + 2 * ds.mldr10_3_std)
                print ('upper')
                lower = (ds.mldr10_3_mean - 2 * ds.mldr10_3_std)
                print ('lower')

                # plot mixed layer depth
                self.axs[0].fill_between(ds.time_counter.dt.dayofyear, 
                                    lower, upper, alpha=0.3, color=colour,
                                    ec=None)
                print ('plt1')
                self.axs[0].plot(ds.time_counter.dt.dayofyear, ds.mldr10_3_mean,                                 lw=1, c=colour)
                print ('plt2')

                print ('siconc')
                l, = self.axs[1].plot(ds.time_counter.dt.dayofyear, 
                            ds.icepres_mean, c=colour, ls='-', lw=1)
                #print ('plt3')

        lines = [l]
        labels= ['SOCHIC ORCA12']

        for ax in self.axs:
        #    #ax.set_xlim([self.ds.time_counter.min(),
        #    #            self.ds.time_counter.max()])
            ax.set_xlim(0,366)
            #major_format = mdates.DateFormatter('%b')
            #ax.xaxis.set_major_formatter(major_format)
        for ax in self.axs[:-1]:
            ax.set_xticks([])

        self.axs[0].set_ylim(-10,160)
        self.axs[0].set_ylabel('mixed layer depth')
        self.axs[1].set_ylabel('mean sea ice presence')

        self.axs[1].set_xlabel('day of year')
        #fig.autofmt_xdate()

        if giddy:
            self.plot_mld_sip_add_giddy()
        if orca:
            lines.append(self.plot_mld_sip_add_orca(colours, years))
            labels.append('global ORCA12')
        if satellite:
            lines.append(self.plot_mld_sip_add_satellite(colours, years))
            labels.append('satellite obs.')
        if argo:
            self.plot_mld_sip_add_argo()
        
        #self.axs[0].legend(lines, labels, fontsize=8)

        plt.savefig('mld_siconc_3year_giddy_and_argo.png', dpi=600)

    def plot_mld_sip_add_giddy(self):
        ''' add giddy to mix layer depth '''

        self.giddy = xr.open_dataset(self.root + 
                    'Giddy_2020/giddy_mld.nc')
        
        self.giddy = self.giddy.groupby('time.dayofyear').mean()
        self.giddy = self.giddy.reindex({'dayofyear': np.arange(1,366)})
        self.axs[0].plot(self.giddy.dayofyear, self.giddy.mld, lw=1, 
                            c='red')

    def plot_mld_sip_add_orca(self, colours, year):
        ''' add orca to sea ice presence '''

        self.orca = xr.open_dataset(self.root + 
                    'NemoOut/ORCA/Stats/ORCA_PATCH_mean_I.nc')
        
        #self.orca = self.orca.assign_coords({'dayofyear':
        #                              self.orca.time_counter.dt.dayofyear})
        #self.orca = self.orca.swap_dims({'time_counter':'dayofyear'})

        #print (year)
        #self.orca = self.orca.sel(time_counter=year[0])

        for i, (year, ds) in enumerate(self.orca.groupby('time_counter.year')): 
            colour = colours[i]
            l, = self.axs[1].plot(ds.time_counter.dt.dayofyear, ds.siconc_mean,
                                  lw=1, c=colour, ls=':')
        return l

    def plot_mld_sip_add_satellite(self, colours, year):
        ''' add satellite to sea ice presence '''

        print (self.root + 'NemoOut/SeaIce/seaice_conc_daily_sh_*mean.nc')
        self.sat = xr.open_mfdataset(self.root + 
                    'NemoOut/SeaIce/seaice_conc_daily_sh_*mean.nc')
        
        #self.sat = self.sat.sel(time=year[0])
        for i, (year, ds) in enumerate(self.sat.groupby('time.year')): 
            #if year == 2014: continue
            colour = colours[i]
            l, =  self.axs[1].plot(ds.time.dt.dayofyear, 
                             ds.icepres_mean, c=colour, ls='--', lw=1)
        return l

    def plot_mld_sip_add_argo(self, clim=True):
        ''' add argo mld '''

        if clim:
            path = '/storage/silver/SO-CHIC/Ryan/Argo/argo_giddy_clim.nc'
            self.argo = xr.open_dataset(path)
            days_in_month_list=[]
            for month in range(1,13):
                _, days_in_month = cal.monthrange(2012, month)
                days_in_month_list.append(days_in_month)

            mid_days = np.array(
                              [int(days_in_month_list[i]/2) for i in range(12)])
            days_in_month = np.array(days_in_month_list)
            self.argo['dayofyear'] = np.cumsum(days_in_month) \
                                                 - days_in_month[0] \
                                                 + mid_days
            upper = self.argo.mld_dt_mean + 2 * self.argo.mld_dt_std
            lower = self.argo.mld_dt_mean - 2 * self.argo.mld_dt_std

            self.axs[0].fill_between(self.argo.dayofyear, 
                                     lower, upper, alpha=0.3, 
                                     color='lightseagreen', ec=None)
            self.axs[0].plot(self.argo.dayofyear,
                             self.argo.mld_dt_mean, lw=1, c='lightseagreen')
 
        else:
            path = '/storage/silver/SO-CHIC/Ryan/Argo/argo_giddy.nc'
            self.argo = xr.open_dataset(path)
            self.axs[0].plot(self.argo.profiledate.dt.dayofyear,
                         self.argo.dt_mld, lw=1, c='lightseagreen')
        
        #self.giddy = self.giddy.reindex({'dayofyear': np.arange(1,366)})

    def plot_mld_sic_bg(self, case, colour):
        
        # alias case
        self.ds = self.cases[case]

        fig, axs = plt.subplots(3,1, figsize=(5.5,5.5))
        
        # calc mixed layer depth stats
        self.area_mean('mldr10_3')
        self.area_std('mldr10_3')
        upper = self.ds.mldr10_3_mean + 2 * self.ds.mldr10_3_std
        lower = self.ds.mldr10_3_mean - 2 * self.ds.mldr10_3_std

        # plot mixed layer depth
        axs[0].fill_between(self.ds.time_counter, lower, upper, alpha=0.2)
        axs[0].plot(self.ds.time_counter, self.ds.mldr10_3_mean, lw=2, 
                    c=colour)

        # sea ice concentration
        self.area_mean('siconc')
        axs[1].plot(self.ds.time_counter, self.ds.siconc_mean, c=colour)


        axs[2].plot(bg_stats.time_counter, bg_stats.dbdx_quant.isel(quantile=1),
                 c=colour, lw=2)
        axs[2].fill_between(bg_stats.time_counter,
                         bg_stats.dbdx_quant.isel(quantile=0),
                         bg_stats.dbdx_quant.isel(quantile=2),
                         color=colour, alpha=0.2)

        #axs[2].plot(bg_stats.time_counter, bg_stats.dbdy_quant.isel(quantile=1),
        #         c='red', lw=2)
        #axs[2].fill_between(bg_stats.time_counter,
        #                 bg_stats.dbdy_quant.isel(quantile=0),
        #                 bg_stats.dbdy_quant.isel(quantile=2),
        #                 color='red', alpha=0.2)
        
        for ax in axs:
            ax.set_xlim([self.ds.time_counter.min(),
                         self.ds.time_counter.max()])
        for ax in axs[:-1]:
            ax.set_xticks([])

        axs[0].set_ylabel('mld')
        axs[1].set_ylabel('siconc')
        axs[2].set_ylabel(r'$|b_x| /  |b_y|$')

        axs[2].set_xlabel('date')
        fig.autofmt_xdate()

        plt.savefig('mld_siconc_bg.png', dpi=300)


    def plot_buoyancy_gradients(self):
        ''' plot buoyancy gradients of model over time '''

        bg_stats = xr.open_dataset(config.data_path() + self.case + 
                                 '/buoyancy_gradient_stats.nc')

        # plot 
        plt.figure()
        plt.plot(bg_stats.time_counter, bg_stats.dbdx_quant.isel(quantile=1),
                 c='black', lw=2)
        plt.fill_between(bg_stats.time_counter,
                         bg_stats.dbdx_quant.isel(quantile=0),
                         bg_stats.dbdx_quant.isel(quantile=2),
                         color='black', alpha=0.2)

        plt.plot(bg_stats.time_counter, bg_stats.dbdy_quant.isel(quantile=1),
                 c='red', lw=2)
        plt.fill_between(bg_stats.time_counter,
                         bg_stats.dbdy_quant.isel(quantile=0),
                         bg_stats.dbdy_quant.isel(quantile=2),
                         color='red', alpha=0.2)

        plt.show()

    def plot_glider_relevant_diagnostics(self):
        '''
        Plot fundamental diagnostics for comparison with glider
        '''

        fig, axs = plt.subplots(6, figsize=(6.5,3)) 
        plt.subplots_adjust()

        


ds = time_series(['EXP04'],[''])
ds.plot_mld_sip(['2012', '2013', '2014'], giddy=True, orca=True, satellite=True,
                                          argo=True)
