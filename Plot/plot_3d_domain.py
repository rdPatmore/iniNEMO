import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import cmocean
import matplotlib
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams.update({'font.size': 8})

class plot_3d_box_and_2d_isopycnals(object):

    def __init__(self):

        self.data_path = config.data_path() + 'EXP10/'

        # plot
        self.fig = plt.figure(figsize=(6.5, 3.5), dpi=1200)
        self.ax = self.fig.add_subplot(projection='3d')

    def plot_box_3d(self):
        plt.subplots_adjust(left=0.00, right=0.78, top=1.05, bottom=0.05)
        cmap = cmocean.cm.solar

        t48_Ro = xr.open_dataarray(self.data_path + 'rossby_number.nc')
        t48_snap = xr.open_dataset(self.data_path + 
                        'SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').votemper
        t48_domain = xr.open_dataset(self.data_path + 'domain_cfg.nc')
        halo=(1%3) + 1
        t = t48_snap.sel(time_counter='2013-02-20 00:00:00', method='nearest')
        t = t.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        Ro = t48_Ro.sel(time_counter='2013-02-20 00:00:00', method='nearest')
        #Ro = Ro.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))

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
        # x face
        p = self.ax.contourf(lon_y, t.isel(y=0).values, dep_y, cmap=cmap,
                        levels=side_levs, zdir='y',
                        offset=lat[0], zorder=1)

        # y face
        pt = self.ax.contourf(t.isel(x=-1).values, lat_x, dep_x, cmap=cmap,
                              levels=side_levs, zdir='x',
                              offset=lon[-1], zorder=1)

        # plot z
        pRo = self.ax.contourf(Ro.nav_lon.values, Ro.nav_lat.values,
                               Ro.isel(depth=0).values,
                               levels=up_levs,
                               cmap=plt.cm.bwr, zdir='z',
                               offset=dep[0])
        # extend matplotlib bug, fixed oct 21, extend='both')

        # x topog
        #p = self.ax.contourf(lon_y, t.isel(y=0).values, dep_y, cmap=cmap,
        #                levels=np.linspace(vmin,vmax,11), zdir='y',
        #                offset=lat[0], zorder=1)
        
        domain = t48_domain.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        z_x = -domain.bathy_meter.isel(t=0,y=0)
        z_y = -domain.bathy_meter.isel(t=0,x=-1)
        self.ax.add_collection3d(plt.fill_between(lon,z_x,-6000, color='grey'),
                                 zs=lat[0], zdir='y')
        self.ax.add_collection3d(plt.fill_between(lat,z_y,-6000, color='dimgrey'),
                                 zs=lon[-1], zdir='x')

        self.ax.set_xlim(lon.min(), lon.max())
        self.ax.set_ylim(lat.min(), lat.max())
        self.ax.set_box_aspect((1, 1, 0.5))

        # remove lines
        self.ax.grid(False)

        # remove background panels
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # move z axis to the left
        tmp_planes = self.ax.zaxis._PLANES
        self.ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
                                  tmp_planes[0], tmp_planes[1],
                                  tmp_planes[4], tmp_planes[5])

        # set axis labels
        self.ax.set_xlabel('Longitude' , fontsize=8)
        self.ax.set_ylabel('Latitude', fontsize=8)
        # rotate label
        self.ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        self.ax.set_zlabel('Depth [m]' , fontsize=8, rotation=90)

        # colour bar
        fig = plt.gcf()
        pos = self.ax.get_position()

        cbar_ax = fig.add_axes([0.70, pos.x0+0.1, 0.02, (pos.y1 - pos.y0)*0.7])
        cbar = fig.colorbar(pt, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.5, 0.5, r'Potential Temperature [$^{\circ}$C]',
                     rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='left')
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        cbar_ax = fig.add_axes([0.85, pos.x0+0.1, 0.02, (pos.y1 - pos.y0)*0.7])
        cbar = fig.colorbar(pRo, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.5, 0.5, r'$\zeta / f$', rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='left')
        cbar.set_ticks((up_levs[0] - up_levs[1]) /2 + up_levs[1::2])
        print(up_levs)
        print(up_levs[1::2])
        print(up_levs[0] + up_levs[1::2])
          
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #cbar.set_ticks([-0.04,-0.02,0, 0.02, 0.04])

        plt.text(0.45, 0.23, 'Maud Rise', transform=self.ax.transAxes,
                     va='center', ha='left', rotation=-30)


        #self.add_frame_3d(self.ax, t)
        #self.ax.set_axis_off()
        plt.savefig('3d_domain.png')

    def add_frame_3d(self, ax, grid):
        ''' add cube frame to 3d plot '''

        from itertools import combinations, product

        # draw cube
        r0 = [grid.nav_lon.min(), grid.nav_lon.max()]
        r1 = [grid.nav_lat.min(), grid.nav_lat.max()]
        r2 = [-grid.deptht.max(), -grid.deptht.min()]
        for i in [0,1]:
            ax.plot([r0[0], r0[0], r0[1], r0[1], r0[0]],
                    [r1[0], r1[1], r1[1], r1[0], r1[0]],
                    [r2[i], r2[i], r2[i], r2[i], r2[i]], color="g",
                                                         zorder=10)
            ax.plot([r0[0], r0[0], r0[1], r0[1], r0[0]],
                    [r1[i], r1[i], r1[i], r1[i], r1[i]],
                    [r2[0], r2[1], r2[1], r2[0], r2[0]], color="g",
                                                         zorder=10)
            ax.plot([r0[i], r0[i], r0[i], r0[i], r0[i]],
                    [r1[0], r1[0], r1[1], r1[1], r1[0]],
                   [r2[0], r2[1], r2[1], r2[0], r2[0]], color="g", 
                                                        zorder=10)

p = plot_3d_box_and_2d_isopycnals()
p.plot_box_3d()
