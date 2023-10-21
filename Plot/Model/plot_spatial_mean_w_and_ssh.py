import config
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.dates as mdates

class time_series(object):

    def __init__(self, cases, title_list):
        self.path = config.data_path()
        self.root = config.root()

        self.cases = cases

        self.ds = {}
        for i, case in enumerate(cases):
            print (self.path + case)
            fn = 'SOCHIC_PATCH_3h_20121209_20130331_grid_'
            fn = 'SOCHIC_PATCH_1h_20120101_20120408_grid_'
            ssh = xr.open_dataset(self.path + case + '/' + fn + 'T.nc').zos
            ssh=ssh.isel(time_counter=slice(-300,-1)).squeeze()
            T = xr.open_dataset(self.path + case + '/' + fn + 'T.nc').votemper
            W = xr.open_dataset(self.path + case + '/' + fn + 'W.nc').wo
            W=W.isel(depthw=10,time_counter=slice(-300,-1)).squeeze()
            T=T.isel(deptht=10,time_counter=slice(-300,-1)).squeeze()
            print (' ')
            print (' ')
            print (' ')
            print (T.values)
            print (' ')
            print (' ')
            print (W.values)
            print (' ')
            print (' ')
            W = W.interp(time_counter=T.time_counter)
                               #chunks={'time_counter':10})
            self.ds[case] = xr.merge([W,T])
            #self.ds[case]['wos'] = self.ds[case]['wo'].isel(depthw=0)
            #self.ds[case]=self.ds[case].isel(depthw=0,deptht=0,
            #                                 time_counter=slice(-300,-1))
            print (self.ds)
            self.ds[case].attrs['title'] = title_list[i]

        #self.cases[year] = self.cases[year].drop_vars('time_instant')
        #self.cases[year] = xr.decode_cf(self.cases[year])
        #self.ds = self.cases[case]
        # buoyancy gradients
        #    if bg:
        #        bg_stats = xr.open_dataset(self.path + case + 
        #                                 '/buoyancy_gradient_stats.nc')
        #        self.cases[case] = xr.merge([bg_stats, self.cases[case]])

    def area_mean(self, case, var):
        ''' mean quantity over depth '''
   
        print (var)
        #self.ds[var + '_mean'] = self.ds[var].mean(['x','y'])
   
        return self.ds[case][var].mean(['x','y'], skipna=True).load()

    def area_prime(self, case, var):
        ''' mean quantity over depth '''
   
        mean = self.ds[case][var].mean(['x','y'], skipna=True).load()
        return mean - self.ds[case][var]

    def plot_w_and_ssh(self):
        ''' '''

        fig, axs = plt.subplots(1,1, figsize=(8.5,2.0))
        plt.subplots_adjust(right=0.82,bottom=0.2,left=0.08,top=0.9)
        for case in self.cases:
            W_prime = self.area_prime(case, 'wo')#.dropna('time_counter')
            T_prime = self.area_prime(case, 'votemper')#.dropna('time_counter')
            heat_flux = (W_prime * T_prime).mean(['x','y']).dropna('time_counter')
            
            axs.plot(heat_flux.time_counter, heat_flux,
                        label=self.ds[case].title)
            #w_mean = self.area_mean(case, 'wo').dropna('time_counter')
            #axs[1].plot(w_mean.time_counter, w_mean,
            #            label=self.ds[case].title)
        axs.set_xlim([heat_flux.time_counter.min(),
                      heat_flux.time_counter.max()])
        #axs[1].set_ylim([w_mean.min(), w_mean.max()])
        #for ax in axs:
        axs.legend(loc='upper left', bbox_to_anchor=(1, 1.0))
        #    ax.set_xlim([ssh_mean.time_counter.isel(time_counter=-300),
        #                 ssh_mean.time_counter.isel(time_counter=-1)])
        #myFmt = DateFormatter("%b %d")
        #axs.xaxis.set_major_formatter(myFmt)
        axs.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axs.set_ylabel(r'$<w^{\prime}T^{\prime}>^{x,y}$, z=16 m')
        plt.savefig('SOCHIC12_heat_flux_8rim.png')
        #plt.show()

#title_list = ['SA3 3 hr','SA3 24 hr','SA3 3 day', 'SA3 10 day']
#ts = time_series(['EXP37','EXP39','EXP40','EXP41'], title_list)

#title_list = ['SA2','SA3','SA3 weak restore',]
#ts = time_series(['EXP23','EXP37','EXP38'], title_list)

title_list = ['SA3','SA3 8rim orlanski', 'SA3 8rim frs']
ts = time_series(['EXP37','EXP42','EXP43'], title_list)

#title_list = ['no spin','short spin','long spin']
#ts = time_series(['EXP23','EXP36','EXP32'], title_list)

#title_list = ['STDts_300','STD_ts100']
#ts = time_series(['EXP24','EXP26'], title_list)

#title_list = ['BESTts_300','BEST_ts50',]
#ts = time_series(['EXP23','EXP25'], title_list)

#title_list = ['SOCHIC12']
#ts = time_series(['EXP23'], title_list)

ts.plot_w_and_ssh()
