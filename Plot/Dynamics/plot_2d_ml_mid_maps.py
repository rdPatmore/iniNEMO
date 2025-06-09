import matplotlib.pyplot as plt
import xarray as xr
import config
import matplotlib
import matplotlib.dates as mdates
import matplotlib.colors as mc
import cmocean
import numpy as np
import iniNEMO.Process.Physics.calc_glider_relevant_diags as grd

matplotlib.rcParams.update({"font.size": 6})

def render_2d_map(fig, ax, da, cmap, title, vmin=None, vmax=None, log=False,
                  cbar=True):
    
    if log:
        kwargs = {'norm':mc.LogNorm(vmin,vmax)}
    else:
        kwargs = {'vmin':vmin, 'vmax':vmax}
    p = ax.pcolor(da.nav_lon, da.nav_lat, da, cmap=cmap, **kwargs)

    if cbar:
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        #cbar.ax.text(8.5, 0.5, title, fontsize=8, rotation=90,
        #             transform=cbar.ax.transAxes, va='center', ha='left')
        cbar.ax.set_ylabel(title)

    return p

def make_highlghted_cmap(cmap_str):
    ''' add green highlight to lower end of given cmap '''

    old_cmap = matplotlib.colormaps[cmap_str].resampled(256)
    newcolours = old_cmap(np.linspace(0, 1, 256))
    green = np.array([102/256, 256/256, 51/256, 1])
    newcolours[:50, :] = green
    new_cmap = mc.ListedColormap(newcolours)
    
    return new_cmap


def cut_rim(ds, rim_size):
    """cut edges from domain"""

    slice_ind = slice(rim_size, -rim_size)
    return ds.isel(x=slice_ind, y=slice_ind)

def get_symetric_limits(da):
    """ return symetric colour bar limits"""

    # get min, max
    vmin = da.min()
    vmax = da.max()

    # set symetric
    abs_max = max(abs(vmin), abs(vmax))
    vmin = -abs_max
    vmax = abs_max
     
    return vmin.values, vmax.values

def add_MIZ_contours(case, ax, grm, quadrant, date, bounds=[0.2,0.8]):
    """ add contours that bound the marginal ice zone """

    ice_path = config.data_path() + case \
             + '/RawOutput/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc'
    ice_conc = xr.open_dataset(ice_path, chunks=-1).siconc

    ice_conc = grm.quadrant_partition(ice_conc, quadrant)
    ice_conc = ice_conc.sel(time_counter=date, method="nearest")
    ice_conc = cut_rim(ice_conc, 10)

    p = ax.contour(ice_conc.nav_lon, ice_conc.nav_lat,
               ice_conc, levels=bounds, linewidths=0.8)

    plt.clabel(p, fontsize=6, inline=True)


def plot_2d_maps(case, date="2012-12-23 12:00:00", quadrant=None):
    """ render 2d maps at mid depth of mixed layer """

    # initialise figure
    fig, axs = plt.subplots(2, 3, figsize=(6.5,3.0))
    plt.subplots_adjust(left=0.1, hspace=0.2, wspace=0.7,
                        top=0.9, bottom=0.15, right=0.85)
    
    # data source
    path = config.data_path() + case \
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"

    # get class for quadrant partitioning
    case = 'EXP10'
    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    grm = grd.glider_relevant_metrics(case, file_id)
                               
    # render temperature
    da = xr.open_dataarray(path + "votemper_ml_mid.nc", chunks=-1)
    da = grm.quadrant_partition(da, quadrant)
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    render_2d_map(fig, axs[0,0], da, cmocean.cm.thermal, "Temperature",
                  vmin=-2, vmax=0)


    # render salinity
    da = xr.open_dataarray(path + "vosaline_ml_mid.nc", chunks=-1)
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    render_2d_map(fig, axs[1,0], da, cmocean.cm.haline, "Salinity", 
                  vmin=33.0, vmax=34.5)

    # render N2
    da = xr.open_dataarray(path + "bn2_ml_mid.nc", chunks=-1)
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    render_2d_map(fig, axs[0,1], da, plt.cm.binary, r"$N^2$",
                  vmin=0, vmax=0.0002)

    # render bg
    da = xr.open_dataarray(path + "bg_mod2_ml_mid.nc", chunks=-1)
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    render_2d_map(fig, axs[1,1], da, plt.cm.binary, r"$|\mathbf{\nabla}b|$",
                  vmin=0, vmax=1e-13)

    path = config.data_path() + case \
         + "/RawOutput/SOCHIC_PATCH_3h_20121209_20130331_"

    # render surface salt flux
    da = xr.open_dataset(path + "grid_T.nc", chunks=-1).sfx
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    vmin, vmax = get_symetric_limits(da)
    render_2d_map(fig, axs[0,2], da, plt.cm.RdBu, "Surface Salt Flux",
                  vmin=vmin, vmax=vmax)

    # render surface heat flux
    da = xr.open_dataset(path + "grid_T.nc", chunks=-1).qt_oce
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    vmin, vmax = get_symetric_limits(da)
    render_2d_map(fig, axs[1,2], da, plt.cm.RdBu, "Surface Heat Flux",
                  vmin=vmin, vmax=vmax)

    for ax in axs.flatten():
        #add_MIZ_contours(case, ax, grm, quadrant, date, bounds=[0.2,0.8])
        ax.set_aspect("equal")
    for ax in axs[0]:
        ax.set_xticklabels([])
    for ax in axs[:,1:].flatten():
        ax.set_yticklabels([])

    # set labels
    for ax in axs[-1]:
        ax.set_xlabel("Longitude")
    for ax in axs[:,0]:
        ax.set_ylabel("Latitude")

    plt.suptitle(date)

    if quadrant:
        append = "_" + quadrant
    else:
        append =""

    plt.savefig("2d_mld_maps_{}{}.png".format(date[:10], append), dpi=1200)

def plot_2d_map_N_M_ml_mid(case, date="2012-12-23 12:00:00", quadrant=None):
    """
    plot snapshot of N and M at the middle of the mixed layer 
    for a given date
    """

    # initialise figure
    fig, axs = plt.subplots(2, 2, figsize=(4.5,3.0))
    plt.subplots_adjust(left=0.1, hspace=0.2, wspace=0.7,
                        top=0.9, bottom=0.15, right=0.85)
    
    # data source
    path = config.data_path() + case \
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"

    # get class for quadrant partitioning
    case = 'EXP10'
    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    grm = grd.glider_relevant_metrics(case, file_id)
                               
    # create cmap with highlighted lower range
    cust_cmap = make_highlghted_cmap('binary')

    # render N2
    da = xr.open_dataarray(path + "bn2_ml_mid.nc", chunks=-1)
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    render_2d_map(fig, axs[0,0], da, plt.cm.binary, r"$N^2$",
                  vmin=0, vmax=0.0002)
    render_2d_map(fig, axs[0,1], da, cust_cmap, r"$N^2$",
                  vmin=1e-14, vmax=1e-3, log=True)

    # render bg
    da = xr.open_dataarray(path + "bg_mod2_ml_mid.nc", chunks=-1)
    da = da.sel(time_counter=date, method="nearest")
    da = grm.quadrant_partition(da, quadrant)
    da = cut_rim(da, 10)
    render_2d_map(fig, axs[1,0], da, plt.cm.binary, r"$|\mathbf{\nabla}b|$",
                  vmin=0, vmax=1e-13)
    render_2d_map(fig, axs[1,1], da, cust_cmap, r"$|\mathbf{\nabla}b|$",
                  vmin=1e-26, vmax=1e-11, log=True)

    for ax in axs.flatten():
        ax.set_aspect('equal')

    if quadrant:
        append = "_" + quadrant
    else:
        append =""


    # save
    plt.savefig("2d_N_M_mld_maps_{}{}.png".format(date[:10], append),
             dpi=1200)

def plot_2d_map_N_M_ml_mid_three_time(case, dates=["2012-12-23 12:00:00"],
                                      quadrant=None):
    """
    plot snapshot of N and M at the middle of the mixed layer 
    for a given date
    """

    # initialise figure
    fig, axs = plt.subplots(2, 3, figsize=(6.5,4.0))
    plt.subplots_adjust(left=0.1, hspace=0.2, wspace=0.3,
                        top=0.9, bottom=0.15, right=0.85)
    
    # data source
    path = config.data_path() + case \
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"

    # get class for quadrant partitioning
    case = 'EXP10'
    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    grm = grd.glider_relevant_metrics(case, file_id)
                               
    # create cmap with highlighted lower range
    cust_cmap = make_highlghted_cmap('binary')

    # render N2
    for i, date in enumerate(dates):
        da = xr.open_dataarray(path + "bn2_ml_mid.nc", chunks=-1)
        da = da.sel(time_counter=date, method="nearest")
        da = grm.quadrant_partition(da, quadrant)
        da = cut_rim(da, 10)
        p = render_2d_map(fig, axs[0,i], da, plt.cm.binary, r"$N^2$ (s$^{-2}$)",
                      vmin=0, vmax=0.0002, cbar=False)
        axs[0,i].set_title(date)

    # add colour bar to last column
    ax = axs[0,-1]
    title = r"N$^2$ (s$^{-2}$)"
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel(title)

    # render bg
    for i, date in enumerate(dates):
        da = xr.open_dataarray(path + "bg_mod2_ml_mid.nc", chunks=-1)
        da = da.sel(time_counter=date, method="nearest")
        da = grm.quadrant_partition(da, quadrant)
        da = cut_rim(da, 10)
        da = da ** 0.5
        p = render_2d_map(fig, axs[1,i], da, plt.cm.binary, r"M$^2$ (s$^{-2}$)",
                      vmin=0, vmax=4e-7, cbar=False)

    # add colour bar to last column
    ax = axs[1,-1]
    title = r"M$^2$ (s$^{-2}$)"
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_ylabel(title)

    for ax in axs.flatten():
        ax.set_aspect('equal')

    if quadrant:
        append = "_" + quadrant
    else:
        append =""

    # set labels
    for ax in axs[-1]:
        ax.set_xlabel("Longitude")
    for ax in axs[:,0]:
        ax.set_ylabel("Latitude")

    # save
    plt.savefig("2d_N_M_mld_maps_three_time_{}{}.png".format(date[:10], append),
             dpi=1200)

def plot_2d_map_N_M_slope(case, date, quadrant):
    """
    plot snapshot of N2, M2 and M2/N2
    """

    # initialise figure
    fig, axs = plt.subplots(1, 3, figsize=(6.5,3.0))
    plt.subplots_adjust(left=0.1, hspace=0.2, wspace=0.7,
                        top=0.9, bottom=0.15, right=0.85)
    
    # data source
    path = config.data_path() + case \
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"

    # get class for quadrant partitioning
    case = 'EXP10'
    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    grm = grd.glider_relevant_metrics(case, file_id)
                               
    # create cmap with highlighted lower range
    cust_cmap = make_highlghted_cmap('binary')

    # render N2
    N2 = xr.open_dataarray(path + "bn2_ml_mid.nc", chunks=-1)
    N2 = N2.sel(time_counter=date, method="nearest")
    N2 = grm.quadrant_partition(N2, quadrant)
    N2 = cut_rim(N2, 10)
    render_2d_map(fig, axs[0], N2, plt.cm.binary, r"$N^2$",
                  vmin=0, vmax=0.0002)

    # render bg
    M4 = xr.open_dataarray(path + "bg_mod2_ml_mid.nc", chunks=-1)
    M4 = M4.sel(time_counter=date, method="nearest")
    M4 = grm.quadrant_partition(M4, quadrant)
    M4 = cut_rim(M4, 10)
    M2 = M4 ** 0.5
    render_2d_map(fig, axs[1], M2, plt.cm.binary, r"$M^2$",
                  vmin=0, vmax=1e-7)

    # render M2/N2
    N2 = cut_rim(N2, 2)
    slope = M2/N2
    render_2d_map(fig, axs[2], slope, plt.cm.viridis, r"$M^2 N^{-2}$",
                  vmin=1e-5, vmax=1e1, log=True)
                  #vmin=0, vmax=np.pi/2, log=False)

    for ax in axs.flatten():
        ax.set_aspect('equal')

    plt.suptitle(date)
        
    if quadrant:
        append = "_" + quadrant
    else:
        append =""

    # save
    #plt.savefig("2d_N_M_slope_maps_{}{}.png".format(date[:10], append),
    #         dpi=1200)
    plt.show()

def plot_2d_map_N_M_slope_three_time(case, dates, quadrant):
    """
    plot snapshot of M2/N2 across three snapshots
    """

    # initialise figure
    fig, axs = plt.subplots(1, 3, figsize=(6.5,3.0))
    plt.subplots_adjust(left=0.1, hspace=0.05, wspace=0.1,
                        top=0.9, bottom=0.1, right=0.85)
    
    # data source
    path = config.data_path() + case \
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"

    # get class for quadrant partitioning
    case = 'EXP10'
    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    grm = grd.glider_relevant_metrics(case, file_id)
                               
    # create cmap with highlighted lower range
    cust_cmap = make_highlghted_cmap('binary')

    for i, date in enumerate(dates):
        # render N2
        N2 = xr.open_dataarray(path + "bn2_ml_mid.nc", chunks=-1)
        N2 = N2.sel(time_counter=date, method="nearest")
        N2 = grm.quadrant_partition(N2, quadrant)
        N2 = cut_rim(N2, 10)

        # render bg
        M4 = xr.open_dataarray(path + "bg_mod2_ml_mid.nc", chunks=-1)
        M4 = M4.sel(time_counter=date, method="nearest")
        M4 = grm.quadrant_partition(M4, quadrant)
        M4 = cut_rim(M4, 10)
        M2 = M4 ** 0.5

        # render M2/N2
        N2 = cut_rim(N2, 2)
        slope = M2/N2
        p = render_2d_map(fig, axs[i], slope, plt.cm.viridis, r"$M^2 N^{-2}$",
                      vmin=1e-4, vmax=1e0, log=True, cbar=False)

        axs[i].set_title(date)

    for ax in axs.flatten():
        ax.set_aspect('equal')

    # add colour bar to last column
    ax = axs[-1]
    title = r"M$^2$N$^{-2}$ (-)"
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical', extend='both')
    cbar.ax.set_ylabel(title)

    # set labels
    for ax in axs[1:]:
        ax.set_yticklabels([])
    for ax in axs:
        ax.set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")

    if quadrant:
        append = "_" + quadrant
    else:
        append =""

    # save
    plt.savefig("2d_N_M_slope_three_time_maps_{}.png".format(append),
             dpi=1200)


if __name__ == "__main__":
    dates = ["2012-12-23 12:00:00",
             "2012-12-24 12:00:00",
             "2012-12-25 12:00:00"]
    plot_2d_map_N_M_slope_three_time("EXP10", dates=dates, quadrant=None)
    #plot_2d_map_N_M_ml_mid_three_time("EXP10", dates=dates, quadrant=None)

