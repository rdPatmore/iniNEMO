import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import cmocean
import cartopy.crs as ccrs
import iniNEMO.Plot.plot_density_ratio as dr

matplotlib.rcParams.update({'font.size': 8})

class paper_hydrog(object):

    def __init__(self, case, subset=None, giddy_method=False):

        self.case = case
        self.subset = subset
        self.giddy_method = giddy_method
        if subset:
            self.subset_var = '_' + subset
        else:
            self.subset_var = ''

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.f_path  = config.data_path() + case + self.file_id 
        self.data_path_old = config.data_path_old() + case + '/'
        self.data_path = config.data_path() + case + '/'

        c1 = '#f18b00'
        self.colours=[c1, 'purple', 'green']# 'navy','turquoise']

    def render_box_3d(self, axs):
        ''' add 3d box of Rossby number and isotherms '''

        cmap = cmocean.cm.solar

        # load data
        t48_Ro = xr.open_dataarray(self.data_path_old + 'rossby_number.nc')
        t48_snap = xr.open_dataset(self.f_path + 'grid_T.nc').votemper
        si = xr.open_dataset(self.f_path + 'icemod.nc').icepres
        t48_domain = xr.open_dataset(self.data_path + 'domain_cfg.nc')

        # restrict data
        halo=10
        time = '2012-12-30 00:00:00'
        t = t48_snap.sel(time_counter=time, method='nearest')
        t = t.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        Ro = t48_Ro.sel(time_counter=time, method='nearest')
        Ro = Ro.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        si = si.sel(time_counter=time, method='nearest')
        si = si.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))

        vmin = t.min()
        vmax = t.max()
        
        lon = t.nav_lon.isel(y=0) 
        lat = t.nav_lat.isel(x=0) 
        dep = -t.deptht

        lon_y, dep_y = np.meshgrid(lon, dep)
        lat_x, dep_x = np.meshgrid(lat, dep)

        side_levs = np.linspace(vmin, vmax, 12)
        up_lin = np.linspace(-0.3, 0.3, 12)
        up_levs = up_lin #- (up_lin[1] - up_lin[0])/2
        ice_levs = np.linspace(0,1,12)

        # x face
        p = axs.contourf(lon_y, t.isel(y=0).values, dep_y, cmap=cmap,
                        levels=side_levs, zdir='y',
                        offset=lat[0], zorder=1)

        # y face
        pt = axs.contourf(t.isel(x=-1).values, lat_x, dep_x, cmap=cmap,
                              levels=side_levs, zdir='x',
                              offset=lon[-1], zorder=1)

        # plot z - Ro
        pRo = axs.contourf(Ro.nav_lon.values, Ro.nav_lat.values,
                               Ro.isel(depth=0).values,
                               levels=up_levs,
                               cmap=cmocean.cm.balance, zdir='z',
                               offset=dep[0])
        # extend matplotlib bug, fixed oct 21, extend='both')

        # x topog
        #p = axs.contourf(lon_y, t.isel(y=0).values, dep_y, cmap=cmap,
        #                levels=np.linspace(vmin,vmax,11), zdir='y',
        #                offset=lat[0], zorder=1)
        
        domain = t48_domain.isel(x=slice(1*halo, -1*halo),
                                 y=slice(1*halo, -1*halo))
        z_x = -domain.bathy_meter.isel(t=0,y=0)
        z_y = -domain.bathy_meter.isel(t=0,x=-1)
        axs.add_collection3d(axs.fill_between(lon,z_x,-6000, color='grey'),
                                 zs=lat[0], zdir='y')
        axs.add_collection3d(axs.fill_between(lat,z_y,-6000, color='dimgrey'),
                                 zs=lon[-1], zdir='x')

        axs.set_xlim(lon.min(), lon.max())
        axs.set_ylim(lat.min(), lat.max())
        axs.set_box_aspect((1, 1, 0.5))

        # remove lines
        axs.grid(False)

        # remove background panels
        axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # move z axis to the left
        tmp_planes = axs.zaxis._PLANES
        axs.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
                              tmp_planes[0], tmp_planes[1],
                              tmp_planes[4], tmp_planes[5])

        # set axis labels
        axs.set_xlabel('Longitude' , fontsize=8)
        axs.set_ylabel('Latitude', fontsize=8)
        # rotate label
        axs.zaxis.set_rotate_label(False)  # disable automatic rotation
        axs.set_zlabel('Depth [m]' , fontsize=8, rotation=90)

        # colour bar
        fig = plt.gcf()
        pos = axs.get_position()

        cbar_ax = fig.add_axes([0.70, pos.y0+0.05, 0.02, (pos.y1 - pos.y0)*0.7])
        cbar = fig.colorbar(pt, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.8, 0.5, r'Potential Temperature [$^{\circ}$C]',
                     rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='left')
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        cbar_ax = fig.add_axes([0.85, pos.y0+0.05, 0.02, (pos.y1 - pos.y0)*0.7])
        cbar = fig.colorbar(pRo, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.8, 0.5, r'$\zeta / f$', rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='left')
        cbar.set_ticks((up_levs[0] - up_levs[1]) /2 + up_levs[1::2])
        print(up_levs)
        print(up_levs[1::2])
        print(up_levs[0] + up_levs[1::2])
          
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #cbar.set_ticks([-0.04,-0.02,0, 0.02, 0.04])

        # add subset lines
        x0 = Ro.nav_lon.min()  - 0.5
        x1 = Ro.nav_lon.min()  - 1.5
        y = [Ro.nav_lat.min(), Ro.nav_lat.max()]
        y_mid = np.mean(y)
        d = dep[0]

        # add subest lines 
        p0, = axs.plot([x0, x0], [y[0], y[1]],
                 [d,  d ], color=self.colours[0])
        p1, = axs.plot([x1, x1], [y_mid+0.20, y[1]],
                 [d,  d ], color=self.colours[1])
        p2, = axs.plot([x1, x1], [y[0], y_mid-0.20],
                 [d,  d ], color=self.colours[2])

        # add text for subset lines
        axs.text(x0, y[0], d, 'full', 'y', transform=axs.transData,
                 va='bottom', ha='center', color=self.colours[0])
        axs.text(x1, y[0], d, 'south', 'y', transform=axs.transData,
                     va='bottom', ha='center', color=self.colours[2])
        axs.text(x1, y_mid, d, 'north', 'y', transform=axs.transData,
                     va='bottom', ha='center', color=self.colours[1])

        # allow lines outside of canvas
        p0.set_clip_on(False)
        p1.set_clip_on(False)
        p2.set_clip_on(False)

        plt.text(0.43, 0.20, 'Maud Rise', transform=axs.transAxes,
                     va='center', ha='left', rotation=-30, fontsize=6)

        plt.text(-0.05, 0.85, '(a)', transform=axs.transAxes,
                 va='top', ha='left', fontsize=8)

    def render_timeseries(self, axs):
        
        def render(ax, ds, var, c='k', ls='-', lw=0.8):
            p = ax.plot(ds.time_counter, ds[var + self.subset_var], 
                        c=c, ls=ls, lw=0.8)
            return p

        def render_density_ratio(ax, var):
            var_mean = var + '_ts_mean'
            lower = self.dr.stats[var_mean].where(self.dr.stats[var_mean] < 1)
            upper = self.dr.stats[var_mean].where(self.dr.stats[var_mean] > 1)
            ax.fill_between(lower.time_counter, lower, 1,
                            edgecolor=None, color='teal')
            ax.fill_between(upper.time_counter, 1, upper,
                            edgecolor=None, color='tab:red')

  
        l = []
        #colours = ['orange', 'purple', 'green']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        c1 = '#f18b00'
        colours=[c1, 'purple', 'green']# 'navy','turquoise']
        ls = ['-', '-', '-']
        for i, subset in enumerate([None, 'north', 'south']):
            lab_str = subset
            if lab_str == None: lab_str = 'all'
            
            # update region
            self.subset = subset
            if subset:
                self.subset_var = '_' + subset
            else:
                self.subset_var = ''
            print ('subset', subset)
            
            # load density ratio functions
            self.dr = dr.plot_buoyancy_ratio(self.case, subset=subset)

            # get stats
            self.dr.get_bg_and_surface_flux_stats(load=True)
            self.dr.get_density_ratio(load=True)
            Tu_compen, Tu_constr, Tu_sal, Tu_tem = self.dr.get_Tu_frac()
            self.dr.get_sea_ice_presence_stats()

            # render 
            kwargs = dict(c=colours[i], ls=ls[i], lw=0.8)
            render(axs[0], self.dr.stats,  'norm_grad_b_ts_mean', **kwargs)
            render(axs[1], self.dr.si*100, 'icepres_ts_mean',     **kwargs)
            p, = render(axs[2], self.dr.stats,  'wfo_ts_mean',         **kwargs)
            l.append(p)
          

        axs[0].legend(l, ['all', 'north', 'south'], loc='upper center',
                       bbox_to_anchor=(0.25, 1.10, 0.5, 0.3), ncol=3)

        # axes formatting
        date_lims = (self.dr.stats.time_counter.min(), 
                     self.dr.stats.time_counter.max())
        
        for ax in axs:
            ax.set_xlim(date_lims)
        for ax in axs[:-1]:
            ax.set_xticklabels([])


        axs[0].set_ylabel(r'$|\nabla b|$' + '\n' +r'[$\times10^{-8}$ s$^{-2}]$')
        axs[1].set_ylabel('sea-ice area' + '\n[%]')
        axs[2].set_ylabel(r'$Q^{fw}$' + '\n' +r'[kg m$^{-2}$ s$^{-1}$]')

        # align labels
        xpos = -0.11  # axes coords
        for ax in axs:
            ax.yaxis.set_label_coords(xpos, 0.5)

        # remove sci nums
        axs[0].yaxis.get_offset_text().set_visible(False)

        # ylims
        axs[1].set_ylim(0,110)

        # date labels
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        axs[-1].set_xlabel('date')

        # letters
        letters = ['(b)','(c)','(d)']
        for i, ax in enumerate(axs):
            ax.text(0.015, 0.96, letters[i], transform=ax.transAxes,
                    va='top', ha='left', fontsize=8)

    def plot_hydrography(self):
        '''
        plot over time - buoyancy gradient mean (n,s,all)
                       - buoyancy gradient std (n,s,all)
                       - fresh water fluxes and wfo (n,s,all)
                       - density ratio
        plot map - Sea ice concentration
                 - Ro
        GridSpec
         a) 1 x 1
         a) 4 x 1
        '''
        
        fig = plt.figure(figsize=(5.5, 5.5), dpi=300)

        lonmin, lonmax = -4,4
        latmin, latmax = -64,-56

        gs0 = gridspec.GridSpec(ncols=1, nrows=1)
        gs1 = gridspec.GridSpec(ncols=1, nrows=3)
        gs0.update(top=1.03, bottom=0.58, left=0.00, right=0.78, wspace=0.05)
        gs1.update(top=0.53, bottom=0.07,  left=0.15, right=0.98, hspace=0.17)

        axs0, axs1 = [], []
        axs0 = fig.add_subplot(gs0[0], projection='3d')
        for i in range(3):
            axs1.append(fig.add_subplot(gs1[i]))

        self.render_box_3d(axs0)
        self.render_timeseries(axs1)

        plt.savefig('hydrography.png', dpi=600)

def plot():
    m = paper_hydrog('EXP10')
    m.plot_hydrography()

plot()
