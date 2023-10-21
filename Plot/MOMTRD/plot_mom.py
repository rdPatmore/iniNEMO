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

        ds[vec + 'trd_lad'] = ds[vec + 'trd_keg'] + ds[vec + 'trd_rvo']

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
        fig = plt.figure(figsize=(6.5,5.5))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=3, nrows=2)
        gs1 = gridspec.GridSpec(ncols=3, nrows=1)
    
        # set frame bounds
        gs0.update(top=0.92, bottom=0.40, left=0.08, wspace=0.1, hspace=0.12,
                   right=0.85)
        gs1.update(top=0.30, bottom=0.08, left=0.08, wspace=0.1,right=0.85)

        # assign axes to lists
        axs0, axs1 = [], []
        for i in range(6):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(3):
            axs1.append(fig.add_subplot(gs1[i]))

        # term selection
        var_list = ['trd_hpg', 'trd_pvo', 'trd_lad',
                    'trd_zad', 'trd_ldf', 'trd_zdf']

        # term labels
        titles = ['pressure grad', 'Coriolis',
                  'lateral\nadvection', 'vertical\nadvection',
                  'lateral\ndiffusion', 'vertical\ndiffusion']


        # plot
        vmin, vmax = -1e-3, 1e-3
        cmap=cmocean.cm.balance
        for i, ax in enumerate(axs0):
            p0 = ax.pcolor(self.ds_2d[self.vec + var_list[i]],
                              vmin=vmin, vmax=vmax, cmap=cmap)
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)

        # get budget closure
        rhs_vars = [self.vec + var for var in var_list]
        sum_rhs = self.ds_2d[rhs_vars].to_array().sum('variable')
        budget_diff = self.ds_2d[self.vec + 'trd_tot'] - sum_rhs

        # plot budget closure
        vmin, vmax = -1e-7, 1e-7
        axs1[0].pcolor(sum_rhs, vmin=vmin, vmax=vmax, cmap=cmap)
        axs1[1].pcolor(self.ds_2d[self.vec + 'trd_tot'],
                      vmin=vmin, vmax=vmax, cmap=cmap)
        p1 = axs1[2].pcolor(budget_diff, vmin=vmin, vmax=vmax, cmap=cmap)

        # budget  labels
        titles = ['Sum of RHS', 'Tendency', 'Difference']
        for i, ax in enumerate(axs1):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)

        # axes labels
        for ax in axs0[:3]:
            ax.set_xticklabels([])
        for ax in axs0[1:3] + axs0[3:] + axs1[1:]:
            ax.set_yticklabels([])
        for ax in axs0[3:] + axs1:
            ax.set_xlabel('x')
        for ax in [axs0[0],axs0[3],axs1[0]]:
            ax.set_ylabel('y')
        for ax in axs0 + axs1:
            ax.set_aspect('equal')

        # colour bar - terms
        pos0 = axs0[2].get_position()
        pos1 = axs0[5].get_position()
        cbar_ax = fig.add_axes([0.87, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'Momentum Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')
        cbar.formatter.set_powerlimits((0, 0))

        # colour bar - budget
        pos = axs1[-1].get_position()
        cbar_ax = fig.add_axes([0.87, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'Momentum Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        # figure title
        ax.text(0.5, 0.99, self.vec.upper() + ' Momentum',
                va='top', ha='center',
                fontsize=10, transform=fig.transFigure)
    
        # save
        plt.savefig(self.case + '_' + self.vec + '_mom_budget_integrated.png')

mom =  mom_trd()
mom.get_mom_terms_depth_integral(case='GYRE_TRD', vec='v')
mom.plot_mom_terms()
