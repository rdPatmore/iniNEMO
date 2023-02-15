import xarray as xr
import matplotlib.pyplot as plt
import config
import matplotlib.dates as mdates
import matplotlib
import cmocean
import numpy as np

matplotlib.rcParams.update({'font.size': 8})

class plot_KE(object):

    def __init__(self, case, file_id):
        self.case = case
        self.preamble = config.data_path() + case + '/' + file_id
        self.path = config.data_path() + case + '/'

    def plot_KE_budget_mld(self):
        ''' plot budget of KE at middepth of the mixed layer '''
        
        # ini figure
        fig, axs = plt.subplots(2, 4, figsize=(5.5,5.5))

        # load and slice
        ds = xr.open_dataset(self.preamble + 'KE_30_budget.nc')
        b_flux = xr.open_dataset(self.preamble + 'b_flux_30.nc').isel(
        time_counter=1)

        # plot
        self.vmin, self.vmax = -5e-6, 5e-6
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
        #render(axs[1,3], b_flux, 'b_flux')
        axs[1,3].pcolor(b_flux['b_flux'],
                  vmin=-5e-6, vmax=5e-6,
                  cmap=self.cmap)#, shading='nearest')

        # titles
        axs[0,0].set_title('hyd p')
        axs[0,1].set_title('ke adv')
        axs[0,2].set_title('zeta adv')
        axs[0,3].set_title('cori')
        axs[1,0].set_title('v adv')
        axs[1,1].set_title('lat diff')
        axs[1,2].set_title('vert diff')
        axs[1,3].set_title('b_flux')

        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])

        plt.savefig(self.case + '_ke_mld_budget_30.png')

    def plot_KE_online_offline_calc_diff(self):
        ''' plot difference between online and offline differnce '''

        fig, axs = plt.subplots(3,9, figsize=(6.5,5.5))
        plt.subplots_adjust(left=0.12, bottom=0.2, top=0.95,right=0.98,
                            hspace=0.05,wspace=0.05)

        # load and slice
        off_ds = xr.open_dataset(self.preamble + 'KE_30_budget.nc')
        time = '2012-12-09 01:00:00'
        off_b_flux = - xr.open_dataset(self.preamble + 'b_flux_30.nc').sel(
        time_counter=time).b_flux
        on_ds = xr.open_dataset(self.preamble + 'KE_30.nc').sel(time_counter=time)

        # set kwargs
        vlim = 1e-3
        cmap=cmocean.cm.balance
        kwargs = dict(vmin=-vlim, vmax=vlim, cmap=cmap)#, shading='nearest')

        # offline calcs
        print (off_ds['trd_hpg'])
        g = 9.81
        g_mod = 9.80665
        off_ds = off_ds * 1026.0
        off_b_flux = off_b_flux * 1026.0

        # get rhs toals
        on_kesum = on_ds.ketrd_hpg + on_ds.ketrd_keg + on_ds.ketrd_rvo +  \
                   on_ds.ketrd_pvo + on_ds.ketrd_zad + on_ds.ketrd_ldf +  on_ds.ketrd_zdf 
        off_kesum = off_ds.trd_hpg + off_ds.trd_keg + off_ds.trd_rvo +  \
                    off_ds.trd_pvo + off_ds.trd_zad + off_ds.trd_zdf 

        axs[0,0].pcolor(off_ds['trd_hpg'], **kwargs)
        axs[0,1].pcolor(off_ds['trd_keg'], **kwargs)
        axs[0,2].pcolor(off_ds['trd_rvo'], **kwargs)
        axs[0,3].pcolor(off_ds['trd_pvo'], **kwargs)
        axs[0,4].pcolor(off_ds['trd_zad'], **kwargs)
        axs[0,5].pcolor(off_ds['trd_ldf'], **kwargs)
        #axs[4,0].pcolor(off_ds['trd_spg2d']-off_ds['trd_hpg'], **kwargs)
        #axs[5,0].pcolor(off_ds['trd_pvo2d'], **kwargs)
        axs[0,6].pcolor(off_ds['trd_zdf'], **kwargs)
        axs[0,7].pcolor(off_b_flux, **kwargs)
        axs[0,8].pcolor(off_kesum, **kwargs)

        # online calcs
        axs[1,0].pcolor(on_ds['ketrd_hpg'], **kwargs)
        axs[1,1].pcolor(on_ds['ketrd_keg'], **kwargs)
        axs[1,2].pcolor(on_ds['ketrd_rvo'], **kwargs)
        axs[1,3].pcolor(on_ds['ketrd_pvo'], **kwargs)
        axs[1,4].pcolor(on_ds['ketrd_zad'], **kwargs)
        axs[1,5].pcolor(on_ds['ketrd_ldf'], **kwargs)
        axs[1,6].pcolor(on_ds['ketrd_zdf'], **kwargs)
        axs[1,7].pcolor(on_ds['ketrd_convP2K'], **kwargs)
        axs[1,8].pcolor(on_kesum, **kwargs)

        # set residual kwargs
        vlim = 5e-4
        kwargs = dict(vmin=-vlim, vmax=vlim, cmap=cmap)#, shading='nearest')

        # render redsidual
        axs[2,0].pcolor(on_ds['ketrd_hpg'] - off_ds['trd_hpg'], **kwargs)
        axs[2,1].pcolor(on_ds['ketrd_keg'] - off_ds['trd_keg'], **kwargs)
        axs[2,2].pcolor(on_ds['ketrd_rvo'] - off_ds['trd_rvo'], **kwargs)
        axs[2,3].pcolor(on_ds['ketrd_pvo'] - off_ds['trd_pvo'], **kwargs)
        axs[2,4].pcolor(on_ds['ketrd_zad'] - off_ds['trd_zad'], **kwargs)
        axs[2,5].pcolor(on_ds['ketrd_ldf'] - off_ds['trd_ldf'], **kwargs)
        axs[2,6].pcolor(on_ds['ketrd_zdf'] - off_ds['trd_zdf'], **kwargs)
        axs[2,7].pcolor(on_ds['ketrd_convP2K'] - off_b_flux, **kwargs)
        p = axs[2,8].pcolor(on_kesum - off_kesum, **kwargs)


        titles = ['pressure grad',
                  'lateral\nadvection\n(non-rota.)',
                  'lateral\nadvection\n(rota.)',
                  'Coriolis',
                  'vertical\nadvection',
                  'lateral\nviscosity',
                  'vertical\nviscosity',
                  'vertical\nbuoyancy flux',
                  'sum of RHS']

        for i, ax in enumerate(axs[0]):
            ax.text(0.5, 1.01, titles[i], va='bottom', ha='center',
                    transform=ax.transAxes, fontsize=6, rotation=0)
        for ax in axs.flatten():
            ax.set_aspect('equal')
        axs[0,0].text(-0.85, 0.5, 'offline', va='center', ha='right',
                transform=axs[0,0].transAxes, fontsize=8, rotation=90)
        axs[1,0].text(-0.85, 0.5, 'online', va='center', ha='right',
                transform=axs[1,0].transAxes, fontsize=8, rotation=90)
        axs[2,0].text(-0.85, 0.5, 'residual', va='center', ha='right',
                transform=axs[2,0].transAxes, fontsize=8, rotation=90)


        for ax in axs[:-1,:].flatten():
            ax.set_xticklabels([])
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[-1,:].flatten():
            ax.set_xlabel('x')
        for ax in axs[:,0].flatten():
            ax.set_ylabel('y')

        pos0 = axs[-1,0].get_position()
        pos1 = axs[-1,-1].get_position()
        cbar_ax = fig.add_axes([pos0.x0, 0.10, pos1.x1 - pos0.x0, 0.02])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -2.5, 'KE Tendency', fontsize=8,
                     rotation=0, transform=cbar.ax.transAxes,
                     va='top', ha='center')

        plt.savefig(self.case + '_ke_online_offline_diff_refine_correct_mom_30.png')

    def plot_KE_residual_mld(self):

        # ini figure
        fig, axs = plt.subplots(1, 3, figsize=(5.5,5.5))

        # load
        ds = xr.open_dataset(self.preamble + 'KE_30_budget.nc')

        # plotting params
        self.vmin, self.vmax = -1e-6, 1e-6
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
        self.vmin, self.vmax = -1e-9, 1e-9
        axs[2].pcolor(resid,
                        vmin=self.vmin, vmax=self.vmax, cmap=self.cmap)

        axs[0].set_title('sum')
        axs[1].set_title('tend')
        axs[2].set_title('resid')

        # save
        plt.savefig(self.case + '_ke_30_budget_resid_offline.png')

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

    def plot_pvo_issues(self, depth_var='deptht'):
        ''' find the missmatch between online KE diags and offline '''

        ke   = xr.open_dataset(self.preamble + 'KE.nc')
        umom = xr.open_dataset(self.preamble + 'momu.nc')
        vmom = xr.open_dataset(self.preamble + 'momv.nc')
        uvel = xr.open_dataset(self.preamble + 'grid_U.nc').uo
        vvel = xr.open_dataset(self.preamble + 'grid_V.nc').vo
        cfg  = xr.open_dataset(self.path + 'domain_cfg.nc').squeeze()
        e3u  = xr.open_dataset(self.preamble + 'grid_U.nc').e3u
        e3v  = xr.open_dataset(self.preamble + 'grid_V.nc').e3v
        e3t  = xr.open_dataset(self.preamble + 'grid_T.nc').e3t


        def cut(ds, depth_var):
            time = '2012-12-09 01:00:00'
            ds = ds.sel({depth_var:30, 'time_counter':time}, method='nearest')
            return ds
        ke   = cut(ke, depth_var='deptht')
        umom = cut(umom, depth_var='depthu')
        vmom = cut(vmom, depth_var='depthv')
        uvel = cut(uvel, depth_var='depthu')
        vvel = cut(vvel, depth_var='depthv')
        e3u = cut(e3u, depth_var='depthu')
        e3v = cut(e3v, depth_var='depthv')
        e3t = cut(e3t, depth_var='deptht')
        print (cfg)

#bu_pvo               (time_counter, deptht, y, x) float32 ...
#    bv_pvo               (time_counter, deptht, y, x) float32 ...
#    r1_bt_pvo            (time_counter, deptht, y, x) float32 ...
#
#   un_pvo               (time_counter, depthu, y, x) float32 ...
#    putrd_pvo            (time_counter, depthu, y, x) float32 ...


        # ini figure
        fig, axs = plt.subplots(7, 3, figsize=(5.5,5.5))
        cmap = cmocean.cm.balance

        def find_lim(var):
            return max(np.abs(var.min()), np.abs(var.max()))

        # u-velocities
        lim = find_lim(umom.un_pvo)
        axs[0,0].pcolor(umom.un_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[0,1].pcolor(uvel, vmin=-lim, vmax=lim, cmap=cmap)
        axs[0,2].pcolor(umom.un_pvo - uvel, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        # v-velocities
        lim = find_lim(vmom.vn_pvo)
        axs[1,0].pcolor(vmom.vn_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[1,1].pcolor(vvel, vmin=-lim, vmax=lim, cmap=cmap)
        axs[1,2].pcolor(vmom.vn_pvo - vvel, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        # u-pvo
        lim = find_lim(umom.utrd_pvo)
        axs[2,0].pcolor(umom.putrd_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[2,1].pcolor(umom.utrd_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[2,2].pcolor(umom.putrd_pvo - umom.putrd_pvo, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        # v-pvo
        lim = find_lim(vmom.vtrd_pvo)
        axs[3,0].pcolor(vmom.pvtrd_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[3,1].pcolor(vmom.vtrd_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[3,2].pcolor(vmom.vtrd_pvo - vmom.pvtrd_pvo, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        bu = cfg.e1u * cfg.e2u * e3u
        bv = cfg.e1v * cfg.e2v * e3v
        bt = cfg.e1t * cfg.e2t * e3t

        # bu 
        lim = find_lim(ke.bu_pvo)
        axs[4,0].pcolor(ke.bu_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[4,1].pcolor(bu, vmin=-lim, vmax=lim, cmap=cmap)
        axs[4,2].pcolor(ke.bu_pvo - bu, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        # bv 
        lim = find_lim(ke.bv_pvo)
        axs[5,0].pcolor(ke.bv_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        axs[5,1].pcolor(bv, vmin=-lim, vmax=lim, cmap=cmap)
        axs[5,2].pcolor(ke.bv_pvo - bv, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        # bt 
        #lim = find_lim(1/ke.r1_bt_pvo)
        #axs[6,0].pcolor(1/ke.r1_bt_pvo, vmin=-lim, vmax=lim, cmap=cmap)
        #axs[6,1].pcolor(bt, vmin=-lim, vmax=lim, cmap=cmap)
        #axs[6,2].pcolor((1/ke.r1_bt_pvo) - bt, vmin=-lim/10, vmax=lim/10, cmap=cmap)

        plt.savefig(self.case + '_pvo_issue_tracking.png')
    
#file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
file_id = 'SOCHIC_PATCH_1h_20121209_20121211_'
ke = plot_KE('TRD00', file_id)
##ke.plot_KE_cori_balance()
#ke.plot_KE_budget_mld()
ke.plot_KE_online_offline_calc_diff()
#ke.plot_KE_residual_mld()
#ke.plot_pvo_issues()
