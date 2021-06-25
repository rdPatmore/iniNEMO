import xarray as xr
import matplotlib.pyplot as plt
import config 

class time_series(object):

    def __init__(self, case):
        self.case = case
        path = config.data_path()
        self.ds = xr.open_mfdataset(path + case + '/SOCHIC_PATCH*.nc',
                                    decode_times=False, compat='override')
        self.ds = self.ds.drop_vars('time_instant')
        self.ds = xr.decode_cf(self.ds)

    def area_mean(self, var):
        ''' mean quantity over depth '''
   
        self.ds[var + '_mean'] = self.ds[var].mean(['x','y'])

    def area_std(self, var):
        ''' mean quantity over depth '''
   
        self.ds[var + '_std'] = self.ds[var].std(['x','y'])
   
    
    def render_panel(self, ax):
        ''' plot the area-mean mixed layer depth '''

        ax.plot(self.ds.time_counter, self.ds[var + '_mean'])

    def plot_mld_sic_bg(self):

        fig, axs = plt.subplots(3,1, figsize=(5.5,5.5))
        
        # calc mixed layer depth stats
        self.area_mean('mldr10_3')
        self.area_std('mldr10_3')
        #axs[0].plot(self.ds.time_counter, self.ds.mldr10_3.stack(z=('x','y')),
        #            lw=0.5, alpha=0.2, c='black')
        upper = self.ds.mldr10_3_mean + 2 * self.ds.mldr10_3_std
        lower = self.ds.mldr10_3_mean - 2 * self.ds.mldr10_3_std

        # plot mixed layer depth
        axs[0].fill_between(self.ds.time_counter, lower, upper, alpha=0.2)
        axs[0].plot(self.ds.time_counter, self.ds.mldr10_3_mean, lw=2, 
                    c='black')

        # sea ice concentration
        self.area_mean('siconc')
        axs[1].plot(self.ds.time_counter, self.ds.siconc_mean)

        # buoyancy gradients
        bg_stats = xr.open_dataset(config.data_path() + self.case + 
                                 '/buoyancy_gradient_stats.nc')

        axs[2].plot(bg_stats.time_counter, bg_stats.dbdx_quant.isel(quantile=1),
                 c='black', lw=2)
        axs[2].fill_between(bg_stats.time_counter,
                         bg_stats.dbdx_quant.isel(quantile=0),
                         bg_stats.dbdx_quant.isel(quantile=2),
                         color='black', alpha=0.2)

        axs[2].plot(bg_stats.time_counter, bg_stats.dbdy_quant.isel(quantile=1),
                 c='red', lw=2)
        axs[2].fill_between(bg_stats.time_counter,
                         bg_stats.dbdy_quant.isel(quantile=0),
                         bg_stats.dbdy_quant.isel(quantile=2),
                         color='red', alpha=0.2)
        
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


ds = time_series('EXP02')
ds.plot_mld_sic_bg()
