import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case, file_id):
        self.case = case
        self.preamble = config.data_path() + case + '/' + file_id

    def plot_MKE_budget_30(self):
        ''' plot budget of KE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))
        plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1,
                            wspace=0.10, hspace=0.11)
        axs[-1,-1].axis('off')

        # load and slice
        ds = xr.open_dataset(self.preamble + 'MKE_30_budget.nc')
        b_flux = xr.open_dataarray(self.preamble + 'b_flux_mean.nc')
        b_flux = b_flux.sel(deptht=30, method='nearest')

        cut=slice(10,-10)
        ds     = ds.isel(x=cut,y=cut)
        b_flux = b_flux.isel(x=cut,y=cut)

        ds['trd_adv'] = ds.trd_keg + ds.trd_rvo
        ds['trd_hpg'] = ds.trd_hpg + b_flux

        # plot
        self.vmin, self.vmax = -2e-6, 2e-6
        self.cmap=cmocean.cm.balance
        def render(ax, ds, var):
            #ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var],
            ax.pcolor(ds[var],
                      vmin=self.vmin, vmax=self.vmax,
                      cmap=self.cmap)#, shading='nearest')

        render(axs[0,0], ds, 'trd_hpg')
        render(axs[0,1], ds, 'trd_adv')
        render(axs[0,2], ds, 'trd_pvo')
        render(axs[0,3], ds, 'trd_zad')
        render(axs[1,0], ds, 'trd_zdf')
        p = axs[1,1].pcolor(-b_flux, vmin=self.vmin, vmax=self.vmax,
                        cmap=self.cmap)
        render(axs[1,2], ds, 'trd_tot')


        # titles
        titles = ['pressure grad',
                  'lateral\n advection ',
                  'Coriolis',               'vertical\nadvection',
                  'vertical\ndiffusion','vertical\nbuoyancy flux',
                  'tendency' ]

        # plot
        for i, ax in enumerate(axs.flatten()[:-1]):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_aspect('equal')

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
        cbar_ax = fig.add_axes([0.86, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(6.0, 0.5, 'MKE Tendency', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        plt.savefig(self.case + '_mke_30_budget.png')

    def plot_MKE_residual_mld(self):

        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(5.5,5.5))

        # load
        ds = xr.open_dataset(self.preamble + 'MKE_mld_budget.nc')

        # plotting params
        self.vmin, self.vmax = -1e-7, 1e-7
        self.cmap = cmocean.cm.balance

        ## sum
        mkesum = ds.trd_hpg + ds.trd_keg + ds.trd_rvo +  \
                 ds.trd_pvo + ds.trd_zad + ds.trd_ldf +  ds.trd_zdf 
        axs[0].pcolor(mkesum,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        # residule
        resid = mkesum - ds.trd_tot
        axs[1].pcolor(ds.trd_tot,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        axs[2].pcolor(resid,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        axs[0].set_title('sum')
        axs[1].set_title('tend')
        axs[2].set_title('resid')

        # save
        plt.savefig(self.case + '_mke_mld_budget_resid.png')

    def plot_MKE_cori_balance(self, snapshot=False):

        # ini figure
        fig, axs = plt.subplots(4, 2, figsize=(5.5,5.5))

        # load and slice
        uf = xr.open_dataset(self.preamble + 'momu_mld.nc').utrd_pvo
        vf = xr.open_dataset(self.preamble + 'momv_mld.nc').vtrd_pvo
        uvel = xr.open_dataset(self.preamble + 'uvel_mld.nc').uo
        vvel = xr.open_dataset(self.preamble + 'vvel_mld.nc').vo

        # regrid to t-pts
        uT_f = (uf + uf.roll(x=1, roll_coords=False)) / 2
        vT_f = (vf + vf.roll(y=1, roll_coords=False)) / 2
        uT_vel = (uvel + uvel.roll(x=1, roll_coords=False)) / 2
        vT_vel = (vvel + vvel.roll(y=1, roll_coords=False)) / 2

        if snapshot:
            uf = uT_f.isel(time_counter=1)
            vf = vT_f.isel(time_counter=1)
            uvel_mean = uvel.isel(time_counter=1)
            vvel_mean = vvel.isel(time_counter=1)

        else:
            # means
            uf = uT_f.mean('time_counter')
            vf = vT_f.mean('time_counter')
            uvel_mean = uvel.mean('time_counter')
            vvel_mean = vvel.mean('time_counter')

        u_MKE = uf * uvel_mean
        v_MKE = vf * vvel_mean

        # plotting params
        self.vmin, self.vmax = -2e-5, 2e-5
        self.cmap = cmocean.cm.balance
        axs[0,0].pcolor(uf, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        axs[0,1].pcolor(vf, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        self.vmin, self.vmax = -5e-1, 5e-1
        axs[1,0].pcolor(uvel_mean, vmin=self.vmin, vmax=self.vmax,
                        cmap=self.cmap)
        axs[1,1].pcolor(vvel_mean, vmin=self.vmin, vmax=self.vmax,
                        cmap=self.cmap)
        self.vmin, self.vmax = -5e-6, 5e-6
        axs[2,0].pcolor(u_MKE, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        axs[2,1].pcolor(-v_MKE, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        self.vmin, self.vmax = -2e-7, 2e-7
        axs[3,0].pcolor(u_MKE+v_MKE, vmin=self.vmin, vmax=self.vmax, 
                       cmap=self.cmap)

        axs[2,0].set_title('u')
        axs[2,1].set_title('v')

        #plt.colorbar(p)

        plt.savefig(self.case + '_mke_mld_cori_balance.png')

    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
ke = plot_KE('TRD00', file_id)
#ke.plot_MKE_cori_balance(snapshot=False)
ke.plot_MKE_budget_30()
#ke.plot_MKE_residual_mld()
