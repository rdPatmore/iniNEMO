import xarray as xr
import config


class plot_3d_box_and_2d_isopycnals(object):

    def __init__(self):
        plt.style.use('~/.config/matplotlib/isobl.mplstyle')

        self.data_path = config.data_path() + 'EXP10/')

        # plot
        self.fig = plt.figure(figsize=(6.5, 2.5), dpi=1200)
        gs0 = gridspec.GridSpec(ncols=1, nrows=1,
                        left=-0.05, right=0.38, hspace=0.1, top=1, bottom=0.0)
        gs1 = gridspec.GridSpec(ncols=3, nrows=1,
                       left=0.4, right=0.99, wspace=0.05, top=0.90, bottom=0.15)

        self.ax0 = self.fig.add_subplot(gs0[0], projection='3d')
        self.ax1 = self.fig.add_subplot(gs1[0])
        self.ax2 = self.fig.add_subplot(gs1[1])
        self.ax3 = self.fig.add_subplot(gs1[2])

        self.axs0 = self.ax0
        self.axs1 = [self.ax1, self.ax2, self.ax3]

 def plot_box_3d(self, case, time=-1):
        Y = 50

        cmap = plt.cm.RdBu_r
        vmin = -0.046
        vmax = 0.046

        t48_Ro = xr.open_dataarray(self.data_path + 'rossby_number.nc')
        t48_snap = xr.open_dataset(self.data_path + 
                        'SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').votemper
        t = t48_snap.sel(time_counter='2013-02-20 00:00:00', method='nearest')
        t = t.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        
        lon = t.nav_lon.isel(y=0) 
        lat = t.nav_lat.isel(x=0) 
        dep = t.deptht

        # x face
        p = self.ax0.contourf(lon, t.isel(y=0), dep, cmap=cmap,
                        levels=np.linspace(vmin,vmax,41), zdir='y',
                        offset=lon[0], zorder=1)

        # y face
        p = self.ax0.contourf(t.isel(y=0), lat, dep, cmap=cmap,
                              levels=np.linspace(vmin,vmax,41), zdir='x',
                              offset=lat[0], zorder=1)

        # plot z
        p = self.ax0.contourf(lon, lat, t.isel(deptht=0), cmap=plt.cm.Greys,
                              levels=np.linspace(0,1,50), zdir='z',
                              offset=dep[0])

        #self.add_frame_3d(self.ax0, t)
        self.ax0.set_axis_off()

   def add_frame_3d(self, ax, grid):
            ''' add cube frame to 3d plot '''

            from itertools import combinations, product

            # draw cube
            r0 = [grid.X.min()-10.5, grid.X.max()+10.5]
            r1 = [grid.Y.min()-10.5, grid.Y.max()+10.5]
            r2 = [grid.Z.min()-1.5, grid.Z.max()+1.5]
            #r1 = [0, 1600.5]
            #r2 = [-407, 0]
            #                      x
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

