import xarray as xr
import matplotlib.pyplot as plt
import cmocean


class mom_trd(object):
    ''' plotting routine for storkey momentum trends '''

    def __init__(self):
        self.path = '/gws/nopw/j04/jmmp/ryapat/MOM_TRD/'
        self.output_str = '/GYRE_1d_00010101_00011230_'

    def plot_mom_terms_horizontal_slice(self, case, vec='u', depth=5):
        ''' plot terms of momentum budget '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))
        plt.subplots_adjust(right=0.78, hspace=0.25, wspace=0.1, top=0.9)

        # load and slice
        path = self.path + case + self.output_str
        try:
            ds = xr.open_dataset(path + vec + 'mom.nc')
        except:
            ds = xr.open_dataset(path + 'mom' + vec + '.nc')
        
        ds = ds.sel({'depth' + vec: depth}, method='nearest') # depth
        ds = ds.isel(time_counter=-1).load()                  # time
            
        # term selection
        var_list = ['trd_hpg', 'trd_keg', 'trd_rvo', 
                    'trd_pvo', 'trd_zad', 'trd_ldf', 'trd_zdf', 'trd_tot']

        # term labels
        titles = ['hyd\npressure grad',
                  'lateral\n advection (KE)', 'lateral\nadvection (zeta)',
                  'Coriolis',               'vertical\nadvection',
                  'lateral\diffusion',
                  'vertical\ndiffusion',     'tendency' ]

        # plot
        vmin, vmax = -1e-6, 1e-6
        cmap=cmocean.cm.balance
        for i, ax in enumerate(axs.flatten()):
            p = ax.pcolor(ds[vec + var_list[i]],
                             vmin=vmin, vmax=vmax, cmap=cmap)
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)

        # axes labels
        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel('x')
        for ax in axs[:,0].flatten():
            ax.set_ylabel('y')

        # colour bar
        pos0 = axs[0,-1].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([0.79, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(9.0, 0.5, 'Momentum Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')
        ax.text(0.5, 0.99, vec + ' momentum', va='top', ha='center',
                fontsize=10, transform=fig.transFigure)
    
        # save
        plt.savefig(case + '_' + vec + '_mom_full_mld_budget.png')

mom =  mom_trd()
mom.plot_mom_terms_horizontal_slice(case='GYRE_TRD', vec='u', depth=5)
