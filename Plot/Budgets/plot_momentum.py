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

    def plot_full_mom_budget_slices(self, vec='u'):
        ''' plot budget of KE - all terms '''

        # ini figure
        fig, axs = plt.subplots(3, 4, figsize=(5.5,5.5))
        plt.subplots_adjust(right=0.78, hspace=0.25, wspace=0.1, top=0.9)

        # load and slice
        ds = xr.open_dataset(self.preamble + vec + 'mom.nc')
        ds = ds.sel({'depth' + vec: 30}, method='nearest') # depth
        ds = ds.isel(time_counter=1).load()            # time
            
        # dyn_spg is added to trd_hpg for dynspg_ts
        var_list = ['trd_hpg', 'trd_keg', 'trd_rvo',
                    'trd_pvo', 'trd_zad', 'trd_ldf', 'trd_zdf',
                    'trd_tot', 'trd_atf', 'trd_udx']


        # titles
        titles = ['hyd\npressure grad',      'surf\npressure grad',
                  'lateral\n advection (KE)', 'lateral\nadvection (zeta)',
                  'Coriolis',               'vertical\nadvection',
                  'horizontal\nviscosity',
                  'vertical\nviscosity',     'tendency',
                  'filter', 'udx or vdy']

        # plot
        vmin, vmax = -1e-4, 1e-4
        cmap=cmocean.cm.balance
        for i, ax in enumerate(axs.flatten()[:-2]):
            p = ax.pcolor(ds[vec + var_list[i]],
                             vmin=vmin, vmax=vmax, cmap=cmap)
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel('x')
        for ax in axs[:,0].flatten():
            ax.set_ylabel('y')

        pos0 = axs[0,-1].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([0.79, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(9.0, 0.5, 'Momentum Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')
        ax.text(0.5, 0.99, vec + ' momentum', va='top', ha='center',
                fontsize=10, transform=fig.transFigure)
    
        plt.savefig(self.case + '_' + vec + '_mom_full_mld_budget.png')


    def plot_mom_budget_slices(self, vec='u'):
        ''' plot budget of KE - important terms only '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))
        plt.subplots_adjust(right=0.78, hspace=0.25, wspace=0.1, top=0.9)

        # load and slice
        ds = xr.open_dataset(self.preamble + vec + 'mom.nc',
                             chunks={"depth"+vec:1})
        print (ds)
        ds = ds.sel({'depth' + vec: 30}, method='nearest') # depth
        ds = ds.isel(time_counter=1).load()            # time

            
        var_list = ['trd_atm2d', 'trd_hpg', 'trd_rvo', 
                    'trd_pvo', 'trd_zad', 'trd_ldf', 'trd_zdf', 'trd_tot']
        #var_list = ['trd_hpg', 'trd_spg', 'trd_keg', 'trd_rvo',
        #            'trd_pvo', 'trd_zad', 'trd_zdf', 'trd_tot']

        # titles
        #titles = ['hyd\npressure grad',      'surf\npressure grad',
        #          'lateral\n advection (KE)', 'lateral\nadvection (zeta)',
        #          'Coriolis',               'vertical\nadvection',
        #          'vertical\ndiffusion',     'tendency' ]
        titles = ['hyd\npressure grad',
                  'lateral\n advection (KE)', 'lateral\nadvection (zeta)',
                  'Coriolis',               'vertical\nadvection',
                  'lateral\ndiffusion',
                  'vertical\ndiffusion',     'tendency' ]

        # plot
        vmin, vmax = -1e-4, 1e-4
        vmin, vmax = -1e-5, 1e-5
        cmap=cmocean.cm.balance
        for i, ax in enumerate(axs.flatten()):
            p = ax.pcolor(ds[vec + var_list[i]],
                             vmin=vmin, vmax=vmax, cmap=cmap)
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel('x')
        for ax in axs[:,0].flatten():
            ax.set_ylabel('y')

        pos0 = axs[0,-1].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([0.79, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(9.0, 0.5, 'Momentum Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')
        ax.text(0.5, 0.99, vec + ' momentum', va='top', ha='center',
                fontsize=10, transform=fig.transFigure)
    
        plt.savefig(self.case + '_' + vec + '_mom_mld_budget.png')

    def plot_mom_residule_budget(self, vec='u'):
        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(5.5,3))
        plt.subplots_adjust(right=0.78)

        # load and slice
        ds = xr.open_dataset(self.preamble + vec + 'mom.nc')
        ds = ds.sel({'depth' + vec: 30}, method='nearest') # depth
        ds = ds.isel(time_counter=1)                       # time

            
        if vec == 'u':
            mom_sum = ds.utrd_hpg + ds.utrd_ldf + ds.utrd_keg + \
                      ds.utrd_rvo + ds.utrd_pvo + ds.utrd_zad + ds.utrd_zdf 
        if vec == 'v':
            # dyn_spg is added to trd_hpg for dynspg_ts
            mom_sum = ds.vtrd_hpg + ds.vtrd_ldf + ds.vtrd_keg + \
                      ds.vtrd_rvo + ds.vtrd_pvo + ds.vtrd_zad + ds.vtrd_zdf

        # plot
        vmin, vmax = -1e-5, 1e-5
        cmap=cmocean.cm.balance
        #axs[0].pcolor(ts, vmin=vmin, vmax=vmax, cmap=cmap)
        p = axs[0].pcolor(ds[vec + 'trd_tot'], vmin=vmin, vmax=vmax, cmap=cmap)
        axs[1].pcolor(mom_sum, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[2].pcolor(mom_sum-ds[vec+'trd_tot'], vmin=vmin, vmax=vmax, cmap=cmap)
        #axs[2].pcolor(mom_sum-ts, vmin=vmin, vmax=vmax, cmap=cmap)
        axs[0].set_title('tendency')
        axs[1].set_title('sum of RHS')
        axs[2].set_title('residual')

        pos = axs[-1].get_position()
        cbar_ax = fig.add_axes([0.79, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(9.0, 0.5, 'Momentum Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        for ax in axs[1:]:
            ax.set_yticklabels([])
        for ax in axs:
            ax.set_xlabel('x')
        axs[0].set_ylabel('y')
        for ax in axs:
            ax.set_aspect('equal')

        plt.savefig(self.case + '_' + vec + '_mom_mld_budget_resid.png')

    
if __name__ == "__main__":
    #file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    #file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
    file_id = 'SOCHIC_PATCH_1h_20121209_20121209_'
    mom = plot_momentum('TRD00', file_id)
    mom.plot_mom_residule_budget(vec='u')
    #mom.plot_mom_residule_budget(vec='v')
