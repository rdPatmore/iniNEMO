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

    def plot_KE_budget_mld(self):
        ''' plot budget of KE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'KE_30_budget.nc')

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

        plt.savefig(self.case + '_ke_mld_budget_30.png')

    def plot_KE_residual_mld(self):

        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(5.5,5.5))

        # load
        ds = xr.open_dataset(self.preamble + 'KE_30_budget.nc')

        # plotting params
        self.vmin, self.vmax = -1e-7, 1e-7
        self.cmap = cmocean.cm.balance

        ## sum
        kesum = ds.trd_hpg + ds.trd_keg + ds.trd_rvo +  \
                ds.trd_pvo + ds.trd_zad + ds.trd_ldf +  ds.trd_zdf 
        axs[0].pcolor(kesum,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        # residule
        resid = kesum - ds.trd_tot
        axs[1].pcolor(ds.trd_tot,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        self.vmin, self.vmax = -1e-10, 1e-10
        axs[2].pcolor(resid,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        axs[0].set_title('sum')
        axs[1].set_title('tend')
        axs[2].set_title('resid')

        # save
        plt.savefig(self.case + '_ke_30_budget_resid.png')

    def plot_KE_cori_balance(self):

        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))

        # load and slice
        uf = xr.open_dataset(self.preamble + 'momu_30.nc').utrd_pvo
        vf = xr.open_dataset(self.preamble + 'momv_30.nc').vtrd_pvo
        uvel = xr.open_dataset(self.preamble + 'uvel_30.nc').uo
        vvel = xr.open_dataset(self.preamble + 'vvel_30.nc').vo

        # get cori correction term
        u_pvo_bta = xr.open_dataarray(self.preamble + 'utrd_pvo_bta_30.nc')
        v_pvo_bta = xr.open_dataarray(self.preamble + 'vtrd_pvo_bta_30.nc')

        # regrid to t-pts
        uT_f = (uf + uf.roll(x=1, roll_coords=False)) / 2
        vT_f = (vf + vf.roll(y=1, roll_coords=False)) / 2
        uT_vel = (uvel + uvel.roll(x=1, roll_coords=False)) / 2
        vT_vel = (vvel + vvel.roll(y=1, roll_coords=False)) / 2

        uf = uT_f.isel(time_counter=1)
        vf = vT_f.isel(time_counter=1)
        uvel_snap = uvel.isel(time_counter=1)
        vvel_snap = vvel.isel(time_counter=1)

        u_KE = uf * uvel_snap
        v_KE = vf * vvel_snap

        u_pvo_bta = u_pvo_bta.squeeze().isel(time_counter=1)
        v_pvo_bta = v_pvo_bta.squeeze().isel(time_counter=1)

        u_cori_corrected = uvel_snap * u_pvo_bta
        v_cori_corrected = vvel_snap * v_pvo_bta
      

        self.vmin, self.vmax = -1e-5, 1e-5
        self.cmap = cmocean.cm.balance

        # plot model Cori
        axs[0,0].pcolor(u_KE, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        axs[1,0].pcolor(v_KE, vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)
        axs[0,0].set_title(r'$u \cdot u_{cori}$ (full term)')
        axs[1,0].set_title(r'$v \cdot v_{cori}$ (full term)')

        # plot gridding error 
        axs[0,1].pcolor(u_cori_corrected, vmin=self.vmin, vmax=self.vmax, 
                       cmap=self.cmap)
        axs[1,1].pcolor(v_cori_corrected, vmin=self.vmin, vmax=self.vmax, 
                       cmap=self.cmap)
        axs[0,1].set_title(r'$u \cdot u_{cori}$ (grid err)')
        axs[1,1].set_title(r'$v \cdot v_{cori}$ (grid err)')

        # plot residual
        axs[0,2].pcolor(u_KE-u_cori_corrected, vmin=self.vmin, vmax=self.vmax, 
                       cmap=self.cmap)
        axs[1,2].pcolor(v_KE-v_cori_corrected, vmin=self.vmin, vmax=self.vmax, 
                       cmap=self.cmap)
        axs[0,2].set_title(r'$u \cdot u_{cori}$ (residual)')
        axs[1,2].set_title(r'$v \cdot v_{cori}$ (residual)')

        # plot sum of residual
        axs[0,3].pcolor((u_KE-u_cori_corrected)
                       +(v_KE-v_cori_corrected), vmin=self.vmin, vmax=self.vmax, 
                       cmap=self.cmap)

        for ax in axs.flatten():
            ax.set_aspect('equal')
        for ax in axs.flatten():
            ax.set_aspect('equal')

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])

        #plt.colorbar(p)

        plt.savefig(self.case + '_ke_mld_cori_balance_30.png')

    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
ke = plot_KE('TRD00', file_id)
ke.plot_KE_cori_balance()
#ke.plot_KE_budget_mld()
#ke.plot_KE_residual_mld()
