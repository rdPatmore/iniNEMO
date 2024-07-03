import matplotlib.pyplot as plt
import xarray as xr
import config
import matplotlib
import matplotlib.dates as mdates

matplotlib.rcParams.update({"font.size": 8})

def plot_time_series(case, ml_mid=False, date_range=[None,None]):
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

    area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
    total_area = area_partition["area_oce"] \
               + area_partition["area_ice"] \
               + area_partition["area_miz"]
    prop_area = area_partition * 100 / total_area

    def render(ax, var, integ_type, title, area):

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

    def render_sea_ice_area(ax, area):
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
        render(axs[4], "bg_mod2", "domain_integ", r"$|\mathbf{\nabla}b|$", prop_area)

    render(axs[2], "mld", "horizontal_integ", "MLD", prop_area)
    # positive down mld
    axs[2].invert_yaxis()
    #render(axs[5], "windsp", "horizontal_integ", r"$U_{10}$")
    render(axs[5], "wfo", "horizontal_integ", r"$Q_{fw}$", prop_area)
    dates = render(axs[6], "taum", "horizontal_integ", r"$|\mathbf{\tau}_s|$", prop_area)
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

if __name__ == "__main__":

    plot_time_series("EXP10", ml_mid=True, date_range=[None,"2013-01-11"])
