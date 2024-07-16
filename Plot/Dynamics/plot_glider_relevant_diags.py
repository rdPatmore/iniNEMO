import matplotlib.pyplot as plt
import xarray as xr
import config
import matplotlib
import matplotlib.dates as mdates
import cmocean
import numpy as np

matplotlib.rcParams.update({"font.size": 8})

class glider_relevant_vars(object):
    """
    a collection of plotting routiens that are relevant for comparison with
    glider data.

    parameters
    ----------
    date_range: set date bounds of time series
    quad: use lat-lon quadrant partitioning 
          args = {"upper_left","upper_right","lower_left","lower_right"} 
    """

    def __init__(self, case, date_range=[None,None], quad=None):
        self.proc_path = config.data_path() + case\
         + "/ProcessedVars/SOCHIC_PATCH_3h_20121209_20130331_"
        self.timeseries_path = config.data_path() + case + "/TimeSeries/"
        self.date_range = date_range
        self.case = case
        self.quad = quad

    def render_1d_time_series(self, ax, var, integ_type, title):
        """ render line plot on axis for given variable """
    
        if self.quad:
            fn_var = var + "_" + quad
        else:
            fn_var = var

        fn_path = self.timeseries_path + fn_var + "_" + integ_type + ".nc"
        ds = xr.open_dataset(fn_path)
        ds[var + "_miz_weighted_mean"] = ds[var + "_miz_weighted_mean"].where(
                                          self.ice_area.area_miz > 1)
        ds[var + "_ice_weighted_mean"] = ds[var + "_ice_weighted_mean"].where(
                                          self.ice_area.area_ice > 1)
        ds = ds.sel(time_counter=slice(self.date_range[0], self.date_range[1]))
        ax.plot(ds.time_counter, ds[var + "_oce_weighted_mean"], label="Oce")
        ax.plot(ds.time_counter, ds[var + "_miz_weighted_mean"], label="MIZ")
        ax.plot(ds.time_counter, ds[var + "_ice_weighted_mean"], label="ICE")
        ax.set_ylabel(title)
    
        # axes formatting
        date_lims = (ds.time_counter.min().values, 
                     ds.time_counter.max().values)
        ax.set_xlim(date_lims)
    
        return ds.time_counter

    def render_2d_time_series(self, fig, axs, var, integ_type, title,
                              cmap=cmocean.cm.thermal,
                              var_append="_weighted_mean"):
        """ render 2d colour map on axis for given variable """
    
        if self.quad:
            fn_var = var + "_" + quad
        else:
            fn_var = var

        # get data
        fn_path = self.proc_path + fn_var + "_" + integ_type + ".nc"
        ds = xr.open_dataset(fn_path)
    
        # check time - hack via overwrite
        if (ds.time_counter != self.ice_area.time_counter).all():
            ds["time_counter"] = self.ice_area.time_counter
    
        # check depth coord
        if "z" in ds.coords:
            ds = ds.rename({"z":"deptht"})
    
        ds[var + "_miz" + var_append] = ds[var + "_miz" + var_append].where(
                                          self.ice_area.area_miz > 1)
        ds[var + "_ice" + var_append] = ds[var + "_ice" + var_append].where(
                                          self.ice_area.area_ice > 1)
        ds = ds.sel(time_counter=slice(self.date_range[0], self.date_range[1]))
    
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
    
    def render_sea_ice_area(self, ax):
        """ render ice, ocean, miz partition on given axis """
    
        # get sea ice area partitions
        date_range_slice = slice(self.date_range[0], self.date_range[1]))
        ice_area = ice_area.sel(time_counter=date_range_slice)
        ax.plot(ice_area.time_counter, self.ice_area["area_oce"], label="Oce")
        ax.plot(ice_area.time_counter, self.ice_area["area_miz"], label="MIZ")
        ax.plot(ice_area.time_counter, self.ice_area["area_ice"], label="ICE")
        ax.set_ylabel("area (%)")
    
        # axes formatting
        date_lims = (area.time_counter.min().values, 
                     area.time_counter.max().values)
        ax.set_xlim(date_lims)
    
    def get_ice_cover_stats(self):
        """ get percentage of area covered by ice, oce and miz """
    
        # get percentage ice cover for full domain weighted by area
        area_partition = xr.open_dataset(path + "area_ice_oce_miz.nc")
        total_area = area_partition["area_oce"] \
                   + area_partition["area_ice"] \
                   + area_partition["area_miz"]
        self.ice_area = area_partition * 100 / total_area
    
    def plot_time_series_core_vars(self, ml_mid=False, date_range=[None,None]):
        """ 
        Plot model time series of:
            - N2
            - Temperature
            - Salinity
            - Mixed Layer Depth
            - Bouyancy Gradients
    
        Each time series has a Sea Ice, Open Ocean and Marginal Ice Zone
        component
        """
    
        # initialise figure
        fig, axs = plt.subplots(8, figsize=(5.5,7))
        plt.subplots_adjust(left=0.2, hspace=0.25, top=0.95, bottom=0.1)
    
    
        if ml_mid:
            integ_str = "ml_mid_horizontal_integ"
            self.render_1d_time_series(axs[0], "votemper", integ_str,
                                       "Temperature")
            self.render_1d_time_series(axs[1], "vosaline", integ_str,
                                       "Salinity")
            self.render_1d_time_series(axs[3], "bn2", integ_str, r"N$^2$")
            self.render_1d_time_series(axs[4], "bg_mod2", integ_str,
                                 r"$|\mathbf{\nabla}b|$")
        else:
            self.render_1d_time_series(axs[0], "votemper", "domain_integ",
                                 "Temperature")
            self.render_1d_time_series(axs[1], "vosaline", "domain_integ",
                                 "Salinity")
            self.render_1d_time_series(axs[3], "N2_mld", "horizontal_integ",
                                 r"N$^2$")
            self.render_1d_time_series(axs[4], "bg_mod2", "domain_integ",
                                 r"$|\mathbf{\nabla}b|$")
    
        self.render_1d_time_series(axs[2], "mld", "horizontal_integ", "MLD")

        # positive down mld
        axs[2].invert_yaxis()

        self.render_1d_time_series(axs[5], "wfo", "horizontal_integ",
                                    r"$Q_{fw}$")
        dates = self.render_1d_time_series(axs[6], "taum", "horizontal_integ",
                                     r"$|\mathbf{\tau}_s|$")
        self.render_sea_ice_area(axs[7])
    
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

        if self.quad:
            append = append + "_" + quad
    
        plt.savefig("glider_relevant_diags_{0}_{1}{2}.pdf".format(d0, d1,
                                                                   append))
    
    def plot_t_s_M_and_N(self):
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
    
        # plot "M" and "N"
        integ_str = "ml_mid_horizontal_integ"
        self.render_1d_time_series(axs[0], "bn2", integ_str, r"N$^2$")
        dates = self.render_1d_time_series(path, axs[1], "bg_mod2", integ_str,
                                      r"$|\mathbf{\nabla}b|$")
    
        self.render_2d_time_series(path, fig, axs[2:5], "votemper",
                              "ml_horizontal_integ",
                              "Temperature", cmocean.cm.thermal)
        self.render_2d_time_series(path, fig, axs[5:8], "vosaline", 
                              "ml_horizontal_integ",
                              "Salinity", cmocean.cm.haline)
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

        if self.quad:
            append = "_" + quad
        else:
            append =""
    
        plt.savefig("density_gradient_controls_{0}_{1}{2}.png".format(d0, d1,
                    append), dpi=600)
    
    def plot_eke_time_series(self, date_range=[None,None]):
        """
        Plot time series of EKE partitioned by Oce, MIZ and Ice zones.
        """
    
        # initialise figure
        fig, axs = plt.subplots(3, figsize=(5.5,7))
        plt.subplots_adjust(left=0.15, right=0.83, 
                            hspace=0.25, top=0.98, bottom=0.07)
    
        # plot EKE
        dates = self.render_2d_time_series(fig, axs, "TKE", "oce_miz_ice",
                              "EKE", plt.cm.Oranges, var_append="")
    
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

        if self.quad:
            append = "_" + quad
        else:
            append =""
    
        # save figure
        plt.savefig("EKE_{0}_{1}{2}.png".format(d0, d1, append), dpi=600)
    

if __name__ == "__main__":

    grv = glider_relevant_vars("EXP10", date_range=[None,"2013-01-11"],
                              quad="lower_right")
    plot_time_series_core_vars("EXP10", ml_mid=True,
                               date_range=[None,"2013-01-11"])
    #plot_t_s_M_and_N("EXP10", date_range=[None,"2013-01-11"])
    #plot_eke_time_series("EXP10", date_range=[None,"2013-01-11"])
