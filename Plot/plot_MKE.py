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

    def plot_MKE_budget_mld(self):
        ''' plot budget of KE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'MKE_mld_budget.nc')

        # plot
        self.vmin, self.vmax = -2e-7, 2e-7
        self.cmap=cmocean.cm.balance
        def render(ax, ds, var):
            #ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var],
            ax.pcolor(ds[var],
                      vmin=self.vmin, vmax=self.vmax,
                      cmap=self.cmap)#, shading='nearest')

        render(axs[0,0], ds, 'trd_hpg')
        render(axs[0,1], ds, 'trd_keg')
        render(axs[0,2], ds, 'trd_rvo')
        render(axs[0,3], ds, 'trd_pvo')
        render(axs[1,0], ds, 'trd_zad')
        render(axs[1,1], ds, 'trd_ldf')
        render(axs[1,2], ds, 'trd_zdf')

        # titles
        axs[0,0].set_title('hyd p')
        axs[0,1].set_title('ke adv')
        axs[0,2].set_title('zeta adv')
        axs[0,3].set_title('cori')
        axs[1,0].set_title('v adv')
        axs[1,1].set_title('lat diff')
        axs[1,2].set_title('vert diff')

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])

        plt.savefig(self.case + '_mke_mld_budget.png')

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
        fig, axs = plt.subplots(3, 2, figsize=(5.5,5.5))

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
        axs[1,0].pcolor(uvel_mean, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        axs[1,1].pcolor(vvel_mean, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        self.vmin, self.vmax = -5e-6, 5e-6
        axs[2,0].pcolor(u_MKE, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        p = axs[2,1].pcolor(-v_MKE, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        axs[2,0].set_title('u')
        axs[2,1].set_title('v')

        plt.colorbar(p)

        plt.savefig(self.case + '_mke_mld_cori_balance.png')

    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
ke = plot_KE('TRD00', file_id)
#ke.plot_MKE_cori_balance(snapshot=True)
#ke.plot_MKE_budget_mld()
ke.plot_MKE_residual_mld()
