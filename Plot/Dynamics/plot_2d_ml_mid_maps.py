import matplotlib.pyplot as plt
import xarray as xr
import config
import matplotlib
import matplotlib.dates as mdates
import cmocean
import numpy as np

matplotlib.rcParams.update({"font.size": 8})

def render_2d_map(ax, da, cmap, vmin=None, vmax=None):
    
    ax.pcolor(da.nav_lon, da.nav_lat, da, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # set labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")

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
     
    return vmin, vmax

def plot_2d_maps(case):
    """ render 2d maps at mid depth of mixed layer """

    # initialise figure
    fig, axs = plt.subplots(3, 2, figsize=(5.5,7))
    plt.subplots_adjust(left=0.2, hspace=0.25, top=0.95, bottom=0.1)
    

    # data source
    path = config.data_path() + case \
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"
    date = "2012-12-23 12:00:00"

    # render temperature
    da = xr.open_dataarray(path + "votemper_ml_mid.nc")
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    render_2d_map(axs[0,0], da, cmocean.cm.thermal)

    # render salinity
    da = xr.open_dataarray(path + "vosaline_ml_mid.nc")
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    render_2d_map(axs[0,1], da, cmocean.cm.haline, vmin=33)

    # render N2
    da = xr.open_dataarray(path + "bn2_ml_mid.nc")
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    render_2d_map(axs[1,0], da, plt.cm.binary)

    # render bg
    da = xr.open_dataarray(path + "bg_mod2_ml_mid.nc")
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    render_2d_map(axs[1,1], da, plt.cm.binary)

    path = config.data_path() + case \
         + "/RawOutput/SOCHIC_PATCH_3h_20121209_20130331_"

    # render surface salt flux
    da = xr.open_dataset(path + "grid_T.nc").sfx
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    vmin, vmax = get_symetric_limits(da)
    render_2d_map(axs[2,0], da, plt.cm.RdBu, vmin=vmin, vmax=vmax)

    # render surface heat flux
    da = xr.open_dataset(path + "grid_T.nc").qt_oce
    da = da.sel(time_counter=date, method="nearest")
    da = cut_rim(da, 10)
    vmin, vmax = get_symetric_limits(da)
    render_2d_map(axs[2,1], da, plt.cm.RdBu, vmin=vmin, vmax=vmax)

    for ax in axs.flatten()[:-2]:
        ax.set_xticklabels([])
    for ax in axs[1]:
        ax.set_yticklabels([])

    plt.show()

if __name__ == "__main__":
    plot_2d_maps("EXP10")

def render_1d_time_series(path, ax, var, integ_type, title, area, date_range):
    """ render line plot on axis for given variable """

    ds = xr.open_dataset(path + var + "_" + integ_type + ".nc")
    ds[var + "_miz_weighted_mean"] = ds[var + "_miz_weighted_mean"].where(
                                      area.area_miz > 1)
    ds[var + "_ice_weighted_mean"] = ds[var + "_ice_weighted_mean"].where(
                                      area.area_ice > 1)
    ds = ds.sel(time_counter=slice(date_range[0], date_range[1]))
    ax.plot(ds.time_counter, ds[var + "_oce_weighted_mean"], label="Oce")
    ax.plot(ds.time_counter, ds[var + "_miz_weighted_mean"], label="MIZ")
    ax.plot(ds.time_counter, ds[var + "_ice_weighted_mean"], label="ICE")
    ax.set_ylabel(title)

    # axes formatting
    date_lims = (ds.time_counter.min().values, 
                 ds.time_counter.max().values)
    ax.set_xlim(date_lims)

    return ds.time_counter

def render_2d_time_series(path, fig, axs, var, integ_type, title, area,
                          date_range,
                          cmap=cmocean.cm.thermal,
                          var_append="_weighted_mean"):
    """ render 2d colour map on axis for given variable """

    ds = xr.open_dataset(path + var + "_" + integ_type + ".nc")

    # check time - hack via overwrite
    if (ds.time_counter != area.time_counter).all():
        ds["time_counter"] = area.time_counter

    # check depth coord
    if "z" in ds.coords:
        ds = ds.rename({"z":"deptht"})

    ds[var + "_miz" + var_append] = ds[var + "_miz" + var_append].where(
                                      area.area_miz > 1)
    ds[var + "_ice" + var_append] = ds[var + "_ice" + var_append].where(
                                      area.area_ice > 1)
    ds = ds.sel(time_counter=slice(date_range[0], date_range[1]))

    # hack for finding min across oce, ice and miz variables
    # it would be better if ice, oce and miz were under a coordinate
    ds_min = ds.min()
    vmin = min(ds_min[var + "_ice" + var_append],
               ds_min[var + "_oce" + var_append],
               ds_min[var + "_miz" + var_append]
               ).values
    ds_max = ds.max()
    vmax = max(ds_max[var + "_ice" + var_append],
               ds_max[var + "_oce" + var_append],
               ds_max[var + "_miz" + var_append]
               ).values

    axs[0].pcolor(ds.time_counter, ds.deptht,
                  ds[var + "_oce" + var_append].T,
                  cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].pcolor(ds.time_counter, ds.deptht,
                  ds[var + "_miz" + var_append].T,
                  cmap=cmap, vmin=vmin, vmax=vmax)
    p = axs[2].pcolor(ds.time_counter, ds.deptht,
                      ds[var + "_ice" + var_append].T,
                      cmap=cmap, vmin=vmin, vmax=vmax)

    # axes formatting
    date_lims = (ds.time_counter.min().values, 
                 ds.time_counter.max().values)

    for ax in axs:
        ax.set_xlim(date_lims)
        ax.set_ylim(0,120)
        ax.invert_yaxis()
        ax.set_ylabel("Depth (m)")

    pos0 = axs[0].get_position()
    pos1 = axs[2].get_position()
    cbar_ax = fig.add_axes([0.85, pos1.y0, 0.02, pos0.y1 - pos1.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.5, 0.5, title, fontsize=8, rotation=90,
                 transform=cbar.ax.transAxes, va='center', ha='left')

    axs[0].text(0.98, 0.1, "Oce", va="bottom", ha="right", rotation=0,
                transform=axs[0].transAxes)
    axs[1].text( 0.98, 0.1, "MIZ", va="bottom", ha="right", rotation=0,
                transform=axs[1].transAxes)
    axs[2].text( 0.98, 0.1, "Ice", va="bottom", ha="right", rotation=0,
                transform=axs[2].transAxes)

    return ds.time_counter

def render_sea_ice_area(ax, area):
    """ render ice, ocean, miz partition on given axis """

    # get sea ice area partitions
    area = area.sel(time_counter=slice(date_range[0], date_range[1]))
    ax.plot(area.time_counter, area["area_oce"], label="Oce")
    ax.plot(area.time_counter, area["area_miz"], label="MIZ")
    ax.plot(area.time_counter, area["area_ice"], label="ICE")
    ax.set_ylabel("area (%)")

    # axes formatting
    date_lims = (area.time_counter.min().values, 
                 area.time_counter.max().values)
    ax.set_xlim(date_lims)


def plot_time_series_core_vars(case, ml_mid=False, date_range=[None,None]):
    """ 
    Plot model time series of:
        - N2
        - Temperature
        - Salinity
        - Mixed Layer Depth
        - Bouyancy Gradients

    Each time series has a Sea Ice, Open Ocean and Marginal Ice Zone component
    """

    # initialise figure
    fig, axs = plt.subplots(8, figsize=(5.5,7))
    plt.subplots_adjust(left=0.2, hspace=0.25, top=0.95, bottom=0.1)

    # data source
    path = config.data_path() + case + "/TimeSeries/"

    # get percentage ice cover for full domain weighted by area
    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
    total_area = area_partition["area_oce"] \
               + area_partition["area_ice"] \
               + area_partition["area_miz"]
    prop_area = area_partition * 100 / total_area


    if ml_mid:
        integ_str = "ml_mid_horizontal_integ"
        render(axs[0], "votemper", integ_str, "Temperature", prop_area)
        render(axs[1], "vosaline", integ_str, "Salinity", prop_area)
        render(axs[3], "bn2", integ_str, r"N$^2$", prop_area)
        render(axs[4], "bg_mod2", integ_str, r"$|\mathbf{\nabla}b|$", prop_area)
    else:
        render(axs[0], "votemper", "domain_integ", "Temperature", prop_area)
        render(axs[1], "vosaline", "domain_integ", "Salinity", prop_area)
        render(axs[3], "N2_mld", "horizontal_integ", r"N$^2$", prop_area)
        render(axs[4], "bg_mod2", "domain_integ", r"$|\mathbf{\nabla}b|$",
               prop_area)

    render(axs[2], "mld", "horizontal_integ", "MLD", prop_area)
    # positive down mld
    axs[2].invert_yaxis()
    #render(axs[5], "windsp", "horizontal_integ", r"$U_{10}$")
    render(axs[5], "wfo", "horizontal_integ", r"$Q_{fw}$", prop_area)
    dates = render(axs[6], "taum", "horizontal_integ", r"$|\mathbf{\tau}_s|$",
                   prop_area)
    render_sea_ice_area(axs[7], prop_area)

    # date labels
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[-1].set_xlabel('Date')

    for ax in axs[:7]:
        ax.set_xticklabels([])

    # align labels
    xpos = -0.18  # axes coords
    for ax in axs:
        ax.yaxis.set_label_coords(xpos, 0.5)

    axs[0].legend()

    if ml_mid:
        append = "_ml_mid"
    else:
        append = ""

    ## get date limits
    d0 = dates.min().dt.strftime("%Y%m%d").values
    d1 = dates.max().dt.strftime("%Y%m%d").values

    plt.savefig("glider_relevant_diags_{0}_{1}{2}.pdf".format(d0, d1,
                                                               append))

def plot_t_s_M_and_N(case, date_range=[None,None]):
    """
    plot time series of:
        - depth profiles of temperature and salinity
        - ml mid vertical stratification (N^2)
        - ml mid horizontal buoynacy gradients
    """

    # initialise figure
    fig, axs = plt.subplots(8, figsize=(5.5,7))
    plt.subplots_adjust(left=0.15, right=0.83, 
                        hspace=0.25, top=0.98, bottom=0.07)

    # data source
    path = config.data_path() + case + "/TimeSeries/"

    # get percentage ice cover for full domain weighted by area
    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
    total_area = area_partition["area_oce"] \
               + area_partition["area_ice"] \
               + area_partition["area_miz"]
    prop_area = area_partition * 100 / total_area

    # plot "M" and "N"
    integ_str = "ml_mid_horizontal_integ"
    render_1d_time_series(path, axs[0], "bn2", integ_str, r"N$^2$", prop_area,
                          date_range)
    dates = render_1d_time_series(path, axs[1], "bg_mod2", integ_str,
                                  r"$|\mathbf{\nabla}b|$", prop_area,
                                  date_range)

    render_2d_time_series(path, fig, axs[2:5], "votemper",
                          "ml_horizontal_integ",
                          "Temperature", prop_area, date_range,
                          cmocean.cm.thermal)
    render_2d_time_series(path, fig, axs[5:8], "vosaline", 
                          "ml_horizontal_integ",
                          "Salinity", prop_area, date_range,
                          cmocean.cm.haline)
    # date labels
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[-1].set_xlabel('Date')

    for ax in axs[:7]:
        ax.set_xticklabels([])

    # align labels
    xpos = -0.18  # axes coords
    for ax in axs:
        ax.yaxis.set_label_coords(xpos, 0.5)

    axs[0].legend()

    ## get date limits
    d0 = dates.min().dt.strftime("%Y%m%d").values
    d1 = dates.max().dt.strftime("%Y%m%d").values

    plt.savefig("density_gradient_controls_{0}_{1}.png".format(d0, d1,),
                dpi=600)

def plot_eke_time_series(case, date_range=[None,None]):
    """
    Plot time series of EKE partitioned by Oce, MIZ and Ice zones.
    """

    # initialise figure
    fig, axs = plt.subplots(3, figsize=(5.5,7))
    plt.subplots_adjust(left=0.15, right=0.83, 
                        hspace=0.25, top=0.98, bottom=0.07)

    # data source
    path = config.data_path() + case + "/TimeSeries/"

    # get percentage ice cover for full domain weighted by area
    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
    total_area = area_partition["area_oce"] \
               + area_partition["area_ice"] \
               + area_partition["area_miz"]
    prop_area = area_partition * 100 / total_area

    # plot "M" and "N"
    path = config.data_path() + case\
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"
    dates = render_2d_time_series(path, fig, axs, "TKE",
                          "oce_miz_ice",
                          "EKE", prop_area, date_range,
                          plt.cm.Oranges, var_append="")

    # date labels
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    axs[-1].set_xlabel('Date')

    for ax in axs[:2]:
        ax.set_xticklabels([])

    ## get date limits
    d0 = dates.min().dt.strftime("%Y%m%d").values
    d1 = dates.max().dt.strftime("%Y%m%d").values

    # save figure
    plt.savefig("EKE_{0}_{1}.png".format(d0, d1,), dpi=600)

