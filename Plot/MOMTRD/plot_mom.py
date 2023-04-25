import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import matplotlib.gridspec as gridspec


class mom_trd(object):
    ''' plotting routine for storkey momentum trends '''

    def __init__(self):
        self.path = '/gws/nopw/j04/jmmp/ryapat/MOM_TRD/'
        self.output_str = '/GYRE_1d_00010101_00011230_'

    def get_mom_terms_horizontal_slice(self, case, vec='u', depth=5):
        ''' plot terms of momentum budget '''

        # load and slice
        path = self.path + case + self.output_str
        try:
            ds = xr.open_dataset(path + vec + 'mom.nc')
        except:
            ds = xr.open_dataset(path + 'mom' + vec + '.nc')
        
        ds = ds.sel({'depth' + vec: depth}, method='nearest') # depth slice
        self.ds_2d = ds.isel(time_counter=-1).load()             # time

    def get_mom_terms_depth_integral(self, case, vec='u'):
        ''' depth integrate momentum budget '''

        self.case = case
        self.vec = vec
        # load and slice
        path = self.path + case + self.output_str
        try:
            ds = xr.open_dataset(path + vec + 'mom.nc')
        except:
            ds = xr.open_dataset(path + 'mom' + vec + '.nc')

        e3 = xr.open_dataset(path + 'grid_' + vec.upper() +'.nc')['e3' + vec]
        e3 = e3.interp(time_counter=ds.time_counter)
        

        ds = ds.drop(['depth' + vec + '_bounds',
                      'time_instant_bounds',
                      'time_counter_bounds'])
        
        ds = (ds * e3).sum('depth' + vec)       # depth integral
        self.ds_2d = ds.isel(time_counter=-2).load()              # time
            
    def plot_mom_terms(self):
        ''' plot 2d momentum budget '''

        # ~~~~~ define figure layout ~~~~ #

        # initialise figure
        fig = plt.figure(figsize=(6.5,6.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=5, nrows=2)
        gs1 = gridspec.GridSpec(ncols=3, nrows=1)
    
        # set frame bounds
        gs0.update(top=0.98, bottom=0.35, left=0.18, wspace=0.1, hspace=0.25,
                   right=0.78)
        gs1.update(top=0.3, bottom=0.02, left=0.18, wspace=0.1,right=0.78)

        # assign axes to lists
        axs0, axs1 = [], []
        for i in range(8):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(3):
            axs1.append(fig.add_subplot(gs1[i]))

        # term selection
        var_list = ['trd_hpg', 'trd_keg', 'trd_rvo',
                    'trd_pvo', 'trd_zad', 'trd_ldf', 'trd_zdf', 'trd_tot']

        # term labels
        titles = ['hyd\npressure grad',
                  'lateral\nadvection (KE)', 'lateral\nadvection (zeta)',
                  'Coriolis',               'vertical\nadvection',
                  'lateral\diffusion',
                  'vertical\ndiffusion',     'tendency' ]

        # plot
        vmin, vmax = -1e-3, 1e-3
        cmap=cmocean.cm.balance
        for i, ax in enumerate(axs0):
            p0 = ax.pcolor(self.ds_2d[self.vec + var_list[i]],
                              vmin=vmin, vmax=vmax, cmap=cmap)
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)

        # get budget closure
        rhs_vars = [self.vec + var for var in var_list[:-1]]
        sum_rhs = self.ds_2d[rhs_vars].to_array().sum('variable')
        budget_diff = self.ds_2d[self.vec + 'trd_tot'] - sum_rhs

        vmin, vmax = -1e-7, 1e-7
        axs1[0].pcolor(sum_rhs, vmin=vmin, vmax=vmax, cmap=cmap)
        axs1[1].pcolor(self.ds_2d[self.vec + 'trd_tot'],
                      vmin=vmin, vmax=vmax, cmap=cmap)
        p1 = axs1[2].pcolor(budget_diff, vmin=vmin, vmax=vmax, cmap=cmap)
#
        # axes labels
        #for ax in axs[:-1,:].flatten():
        #    ax.set_xticklabels([])
        #for ax in axs[:,1:].flatten():
        #    ax.set_yticklabels([])
        #for ax in axs[-1,:].flatten():
        #    ax.set_xlabel('x')
        #for ax in axs[:,0].flatten():
        #    ax.set_ylabel('y')

        # colour bar
        #pos0 = axs[0,-1].get_position()
        #pos1 = axs[-1,-1].get_position()
        #cbar_ax = fig.add_axes([0.79, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        #cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        #cbar.ax.text(9.0, 0.5, 'Momentum Tendency', fontsize=8,
        #             rotation=90, transform=cbar.ax.transAxes,
        #             va='center', ha='right')
        #ax.text(0.5, 0.99, self.vec + ' momentum', va='top', ha='center',
        #        fontsize=10, transform=fig.transFigure)
    
        # save
        plt.savefig(self.case + '_' + self.vec + '_mom_budget_integrated.png')

mom =  mom_trd()
mom.get_mom_terms_depth_integral(case='GYRE_TRD', vec='u')
mom.plot_mom_terms()
