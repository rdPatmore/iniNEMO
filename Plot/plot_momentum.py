import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean

matplotlib.rcParams.update({'font.size': 8})

class plot_momentum(object):

    def __init__(self, case, file_id):
        self.case = case
        self.preamble = config.data_path() + case + '/' + file_id

    def plot_mom_budget_slices(self, vec='u'):
        ''' plot budget of KE '''
        
        # ini figure
        fig, axs = plt.subplots(4, 4, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'mom' + vec + '.nc')
        print (ds)
        ds = ds.sel({'depth' + vec: 30}, method='nearest') # depth
        ds = ds.isel(time_counter=1).load()            # time

            
        if vec == 'u':
            missing = 'trd_udx'
        if vec == 'v':
            missing = 'trd_vdy'
        var_list = ['trd_hpg', 'trd_spg', 'trd_spgexp', 'trd_spgflt',
                    'trd_keg', 'trd_rvo', 'trd_pvo', 'trd_zad',
                    missing, 'trd_ldf', 'trd_zdf', 'trd_tau',
                    'trd_bfr', 'trd_bfri', 'trd_tot', 'trd_atf']

        # plot
        vmin, vmax = -1e-5, 1e-5
        cmap=cmocean.cm.balance
        for i, ax in enumerate(axs.flatten()):
            ax.pcolor(ds[vec + var_list[i]], vmin=vmin, vmax=vmax, cmap=cmap)

        # sum
        #kesum = ds.ketrd_hpg + ds.ketrd_spg + ds.ketrd_keg + ds.ketrd_rvo +  \
        #        ds.ketrd_pvo + ds.ketrd_zad + \
        #        ds.ketrd_zdf + ds.ketrd_tau + \
        #        ds.ketrd_atf + ds.ketrd_convP2K
        #axs[3,2].pcolor(kesum, vmin=vmin, vmax=vmax, cmap=cmap)

        # residule
        #resid = kesum - trd_tot
        #axs[3,3].pcolor(resid, vmin=vmin, vmax=vmax, cmap=cmap)


        # titles
        axs[0,0].set_title('hyd p')
        axs[0,1].set_title('surf p')
        axs[0,2].set_title('surf p exp')
        axs[0,3].set_title('surf p imp')
        axs[1,0].set_title('ke grad')
        axs[1,1].set_title('zeta')
        axs[1,2].set_title('cori')
        axs[1,3].set_title('vadv')
        axs[2,0].set_title('udx')
        axs[2,1].set_title('lat diff')
        axs[2,2].set_title('z diff')
        axs[2,3].set_title('tau')
        axs[3,0].set_title('drag')
        axs[3,1].set_title('drag impl')
        axs[3,2].set_title('tot')
        axs[3,3].set_title('filter')
        #axs[3,4].set_title('residule')

        plt.savefig(self.case + '_' + vec + '_mom_mld_budget.png')

    def plot_mom_residule_budget(self, vec='u'):
        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'mom' + vec + '.nc')
        print (ds)
        ds = ds.sel({'depth' + vec: 30}, method='nearest') # depth
        ds = ds.isel(time_counter=1)             # time

            
        if vec == 'u':
            mom_sum = ds.utrd_hpg + ds.utrd_spg + ds.utrd_keg + \
                      ds.utrd_rvo + ds.utrd_pvo + ds.utrd_zad
        if vec == 'v':
            mom_sum = ds.vtrd_hpg + ds.vtrd_spg + ds.vtrd_keg + \
                      ds.vtrd_rvo + ds.vtrd_pvo + ds.vtrd_zad

        # plot
        vmin, vmax = -1e-5, 1e-5
        cmap=cmocean.cm.balance
        axs[0].pcolor(ds[vec + 'trd_tot'], vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].pcolor(mom_sum, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].pcolor(mom_sum-ds[vec+'trd_tot'], vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title('tot')
        axs[1].set_title('sum')
        axs[2].set_title('residule')
        plt.savefig(self.case + '_' + vec + '_mom_mld_budget_resid.png')

    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
mom = plot_momentum('EXP90', file_id)
mom.plot_mom_residule_budget(vec='u')
mom.plot_mom_residule_budget(vec='v')
mom.plot_mom_budget_slices(vec='u')
mom.plot_mom_budget_slices(vec='v')

