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

def render_2d_map(fig, ax, da, cmap, title, vmin=None, vmax=None, log=False):
    
    if log:
        kwargs = {'norm':mc.LogNorm(vmin,vmax)}
    else:
        kwargs = {'vmin':vmin, 'vmax':vmax}
    p = ax.pcolor(da.nav_lon, da.nav_lat, da, cmap=cmap, **kwargs)

    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.01, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    #cbar.ax.text(8.5, 0.5, title, fontsize=8, rotation=90,
    #             transform=cbar.ax.transAxes, va='center', ha='left')
    cbar.ax.set_ylabel(title)

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

    plt.suptitle(date)
        
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
    fig, axs = plt.subplots(2, 3, figsize=(4.5,3.0))
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
    for i, date in enumerate(dates):
        da = xr.open_dataarray(path + "bn2_ml_mid.nc", chunks=-1)
        da = da.sel(time_counter=date, method="nearest")
        da = grm.quadrant_partition(da, quadrant)
        da = cut_rim(da, 10)
        render_2d_map(fig, axs[0,i], da, plt.cm.binary, r"$N^2$ (s$^{-2}$)",
                      vmin=0, vmax=0.0002)

    # render bg
    for i, date in enumerate(dates):
        da = xr.open_dataarray(path + "bg_mod2_ml_mid.nc", chunks=-1)
        da = da.sel(time_counter=date, method="nearest")
        da = grm.quadrant_partition(da, quadrant)
        da = cut_rim(da, 10)
        render_2d_map(fig, axs[1,i], da, plt.cm.binary, r"M$^2$ (s$^{-2}$)",
                      vmin=0, vmax=4e-7)

    for ax in axs.flatten():
        ax.set_aspect('equal')

    plt.suptitle(date)
        
    if quadrant:
        append = "_" + quadrant
    else:
        append =""

    # save
    plt.savefig("2d_N_M_mld_maps_three_time_{}{}.png".format(date[:10], append),
             dpi=1200)

def plot_2d_map_N_M_slope(case, date, quadrant):
    """
    plot snapshot of N2, M2 and M2/N2
    """

    # initialise figure
    fig, axs = plt.subplots(1, 3, figsize=(4.5,3.0))
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
    slope = np.arctan(M2/N2)
    render_2d_map(fig, axs[2], slope, plt.cm.RdBu_r, r"$M^2 N^{-2}$",
                  vmin=0, vmax=np.pi/2)

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


if __name__ == "__main__":
    #plot_2d_map_N_M_slope("EXP10", date="2012-12-24 12:00:00", quadrant=None)
    dates = ["2012-12-23 12:00:00",
             "2012-12-24 12:00:00",
             "2012-12-25 12:00:00"]
    plot_2d_map_N_M_ml_mid_three_time("EXP10", dates=dates, quadrant=None)

#def render_1d_time_series(path, ax, var, integ_type, title, area, date_range,
#                          vlims=None):
#    """ render line plot on axis for given variable """
#
#    ds = xr.open_dataset(path + var + "_" + integ_type + ".nc")
#    ds[var + "_miz_weighted_mean"] = ds[var + "_miz_weighted_mean"].where(
#                                      area.area_miz > 1)
#    ds[var + "_ice_weighted_mean"] = ds[var + "_ice_weighted_mean"].where(
#                                      area.area_ice > 1)
#    ds = ds.sel(time_counter=slice(date_range[0], date_range[1]))
#    ax.plot(ds.time_counter, ds[var + "_oce_weighted_mean"], label="Oce")
#    ax.plot(ds.time_counter, ds[var + "_miz_weighted_mean"], label="MIZ")
#    ax.plot(ds.time_counter, ds[var + "_ice_weighted_mean"], label="ICE")
#    ax.set_ylabel(title)
#
#    # axes formatting
#    date_lims = (ds.time_counter.min().values, 
#                 ds.time_counter.max().values)
#    ax.set_xlim(date_lims)
#    if vlims:
#        ax.set_ylim(vlims)
#
#    return ds.time_counter
#
#def render_2d_time_series(path, fig, axs, var, integ_type, title, area,
#                          date_range,
#                          cmap=cmocean.cm.thermal,
#                          var_append="_weighted_mean", vlims=None):
#    """ render 2d colour map on axis for given variable """
#
#    ds = xr.open_dataset(path + var + "_" + integ_type + ".nc")
#
#    # check time - hack via overwrite
#    if (ds.time_counter != area.time_counter).all():
#        ds["time_counter"] = area.time_counter
#
#    # check depth coord
#    if "z" in ds.coords:
#        ds = ds.rename({"z":"deptht"})
#
#    ds[var + "_miz" + var_append] = ds[var + "_miz" + var_append].where(
#                                      area.area_miz > 1)
#    ds[var + "_ice" + var_append] = ds[var + "_ice" + var_append].where(
#                                      area.area_ice > 1)
#    ds = ds.sel(time_counter=slice(date_range[0], date_range[1]))
#
#    # set value limits
#    if vlims:
#        vmin, vmax = vlims[0], vlims[1]
#    else:
#        # hack for finding min across oce, ice and miz variables
#        # it would be better if ice, oce and miz were under a coordinate
#        ds_min = ds.min()
#        vmin = min(ds_min[var + "_ice" + var_append],
#                   ds_min[var + "_oce" + var_append],
#                   ds_min[var + "_miz" + var_append]
#                   ).values
#        ds_max = ds.max()
#        vmax = max(ds_max[var + "_ice" + var_append],
#                   ds_max[var + "_oce" + var_append],
#                   ds_max[var + "_miz" + var_append]
#                   ).values
#
#    axs[0].pcolor(ds.time_counter, ds.deptht,
#                  ds[var + "_oce" + var_append].T,
#                  cmap=cmap, vmin=vmin, vmax=vmax)
#    axs[1].pcolor(ds.time_counter, ds.deptht,
#                  ds[var + "_miz" + var_append].T,
#                  cmap=cmap, vmin=vmin, vmax=vmax)
#    p = axs[2].pcolor(ds.time_counter, ds.deptht,
#                      ds[var + "_ice" + var_append].T,
#                      cmap=cmap, vmin=vmin, vmax=vmax)
#
#    # axes formatting
#    date_lims = (ds.time_counter.min().values, 
#                 ds.time_counter.max().values)
#
#    for ax in axs:
#        ax.set_xlim(date_lims)
#        ax.set_ylim(0,120)
#        ax.invert_yaxis()
#        ax.set_ylabel("Depth (m)")
#
#    pos0 = axs[0].get_position()
#    pos1 = axs[2].get_position()
#    cbar_ax = fig.add_axes([0.85, pos1.y0, 0.02, pos0.y1 - pos1.y0])
#    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
#    cbar.ax.text(4.5, 0.5, title, fontsize=8, rotation=90,
#                 transform=cbar.ax.transAxes, va='center', ha='left')
#
#    axs[0].text(0.98, 0.1, "Oce", va="bottom", ha="right", rotation=0,
#                transform=axs[0].transAxes)
#    axs[1].text( 0.98, 0.1, "MIZ", va="bottom", ha="right", rotation=0,
#                transform=axs[1].transAxes)
#    axs[2].text( 0.98, 0.1, "Ice", va="bottom", ha="right", rotation=0,
#                transform=axs[2].transAxes)
#
#    return ds.time_counter
#
#def render_sea_ice_area(ax, area):
#    """ render ice, ocean, miz partition on given axis """
#
#    # get sea ice area partitions
#    area = area.sel(time_counter=slice(date_range[0], date_range[1]))
#    ax.plot(area.time_counter, area["area_oce"], label="Oce")
#    ax.plot(area.time_counter, area["area_miz"], label="MIZ")
#    ax.plot(area.time_counter, area["area_ice"], label="ICE")
#    ax.set_ylabel("area (%)")
#
#    # axes formatting
#    date_lims = (area.time_counter.min().values, 
#                 area.time_counter.max().values)
#    ax.set_xlim(date_lims)
#
#
#def plot_time_series_core_vars(case, ml_mid=False, date_range=[None,None]):
#    """ 
#    Plot model time series of:
#        - N2
#        - Temperature
#        - Salinity
#        - Mixed Layer Depth
#        - Bouyancy Gradients
#
#    Each time series has a Sea Ice, Open Ocean and Marginal Ice Zone component
#    """
#
#    # initialise figure
#    fig, axs = plt.subplots(8, figsize=(5.5,7))
#    plt.subplots_adjust(left=0.2, hspace=0.25, top=0.95, bottom=0.1)
#
#    # data source
#    path = config.data_path() + case + "/TimeSeries/"
#
#    # get percentage ice cover for full domain weighted by area
#    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
#    total_area = area_partition["area_oce"] \
#               + area_partition["area_ice"] \
#               + area_partition["area_miz"]
#    prop_area = area_partition * 100 / total_area
#
#
#    if ml_mid:
#        integ_str = "ml_mid_horizontal_integ"
#        render(axs[0], "votemper", integ_str, "Temperature", prop_area)
#        render(axs[1], "vosaline", integ_str, "Salinity", prop_area)
#        render(axs[3], "bn2", integ_str, r"N$^2$", prop_area)
#        render(axs[4], "bg_mod2", integ_str, r"$|\mathbf{\nabla}b|$", prop_area)
#    else:
#        render(axs[0], "votemper", "domain_integ", "Temperature", prop_area)
#        render(axs[1], "vosaline", "domain_integ", "Salinity", prop_area)
#        render(axs[3], "N2_mld", "horizontal_integ", r"N$^2$", prop_area)
#        render(axs[4], "bg_mod2", "domain_integ", r"$|\mathbf{\nabla}b|$",
#               prop_area)
#
#    render(axs[2], "mld", "horizontal_integ", "MLD", prop_area)
#    # positive down mld
#    axs[2].invert_yaxis()
#    #render(axs[5], "windsp", "horizontal_integ", r"$U_{10}$")
#    render(axs[5], "wfo", "horizontal_integ", r"$Q_{fw}$", prop_area)
#    dates = render(axs[6], "taum", "horizontal_integ", r"$|\mathbf{\tau}_s|$",
#                   prop_area)
#    render_sea_ice_area(axs[7], prop_area)
#
#    # date labels
#    for ax in axs:
#        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
#    axs[-1].set_xlabel('Date')
#
#    for ax in axs[:7]:
#        ax.set_xticklabels([])
#
#    # align labels
#    xpos = -0.18  # axes coords
#    for ax in axs:
#        ax.yaxis.set_label_coords(xpos, 0.5)
#
#    axs[0].legend()
#
#    if ml_mid:
#        append = "_ml_mid"
#    else:
#        append = ""
#
#    ## get date limits
#    d0 = dates.min().dt.strftime("%Y%m%d").values
#    d1 = dates.max().dt.strftime("%Y%m%d").values
#
#    plt.savefig("glider_relevant_diags_{0}_{1}{2}.pdf".format(d0, d1,
#                                                               append))
#
#def plot_t_s_M_and_N(case, date_range=[None,None]):
#    """
#    plot time series of:
#        - depth profiles of temperature and salinity
#        - ml mid vertical stratification (N^2)
#        - ml mid horizontal buoynacy gradients
#    """
#
#    # initialise figure
#    fig, axs = plt.subplots(8, figsize=(5.5,7))
#    plt.subplots_adjust(left=0.15, right=0.83, 
#                        hspace=0.25, top=0.98, bottom=0.07)
#
#    # data source
#    path = config.data_path() + case + "/TimeSeries/"
#
#    # get percentage ice cover for full domain weighted by area
#    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
#    total_area = area_partition["area_oce"] \
#               + area_partition["area_ice"] \
#               + area_partition["area_miz"]
#    prop_area = area_partition * 100 / total_area
#
#    # plot "M" and "N"
#    integ_str = "ml_mid_horizontal_integ"
#    render_1d_time_series(path, axs[0], "bn2", integ_str, r"N$^2$ (s$^{-2}$)",
#                          prop_area, date_range, vlims=[0,0.0002])
#    dates = render_1d_time_series(path, axs[1], "bg_mod2", integ_str,
#                                  r"$|\mathbf{\nabla}b|$ (s$^{-2}$)", prop_area,
#                                  date_range, vlims=[0,2.1e-14])
#
#    render_2d_time_series(path, fig, axs[2:5], "votemper",
#                          "ml_horizontal_integ",
#                          r"Temperature ($^{\circ}$C)", prop_area, date_range,
#                          cmocean.cm.thermal, vlims=[-1.9,-0.1])
#    render_2d_time_series(path, fig, axs[5:8], "vosaline", 
#                          "ml_horizontal_integ",
#                          r"Salinity ($10^3$)", prop_area, date_range,
#                          cmocean.cm.haline, vlims=[33.3,34.4])
#    # date labels
#    for ax in axs:
#        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
#    axs[-1].set_xlabel('Date')
#
#    for ax in axs[:7]:
#        ax.set_xticklabels([])
#
#    # align labels
#    xpos = -0.18  # axes coords
#    for ax in axs:
#        ax.yaxis.set_label_coords(xpos, 0.5)
#
#    axs[0].legend()
#
#    ## get date limits
#    d0 = dates.min().dt.strftime("%Y%m%d").values
#    d1 = dates.max().dt.strftime("%Y%m%d").values
#
#    plt.savefig("density_gradient_controls_{0}_{1}.png".format(d0, d1),
#                dpi=600)
#
#def plot_eke_time_series(case, date_range=[None,None]):
#    """
#    Plot time series of EKE partitioned by Oce, MIZ and Ice zones.
#    """
#
#    # initialise figure
#    fig, axs = plt.subplots(3, figsize=(5.5,7))
#    plt.subplots_adjust(left=0.15, right=0.83, 
#                        hspace=0.25, top=0.98, bottom=0.07)
#
#    # data source
#    path = config.data_path() + case + "/TimeSeries/"
#
#    # get percentage ice cover for full domain weighted by area
#    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
#    total_area = area_partition["area_oce"] \
#               + area_partition["area_ice"] \
#               + area_partition["area_miz"]
#    prop_area = area_partition * 100 / total_area
#
#    # plot "M" and "N"
#    path = config.data_path() + case\
#         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"
#    dates = render_2d_time_series(path, fig, axs, "TKE",
#                          "oce_miz_ice",
#                          "EKE", prop_area, date_range,
#                          plt.cm.Oranges, var_append="")
#
#    # date labels
#    for ax in axs:
#        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
#    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
#    axs[-1].set_xlabel('Date')
#
#    for ax in axs[:2]:
#        ax.set_xticklabels([])
#
#    ## get date limits
#    d0 = dates.min().dt.strftime("%Y%m%d").values
#    d1 = dates.max().dt.strftime("%Y%m%d").values
#
#    # save figure
#    plt.savefig("EKE_{0}_{1}.png".format(d0, d1,), dpi=600)
#
