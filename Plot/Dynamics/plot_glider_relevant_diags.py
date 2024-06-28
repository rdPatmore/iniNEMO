import matplotlib.pyplot as plt
import xarray as xr
import config
import matplotlib

matplotlib.rcParams.update({"font.size": 8})

def plot_time_series(case, ml_mid=False):
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
    plt.subplots_adjust()

    # data source
    path = config.data_path() + case + "/TimeSeries/"

    def render(ax, var, integ_type, title):

        ds = xr.open_dataset(path + var + "_" + integ_type + ".nc")
        ax.plot(ds.time_counter, ds[var + "_oce_weighted_mean", label="Oce")
        ax.plot(ds.time_counter, ds[var + "_miz_weighted_mean", label="MIZ")
        ax.plot(ds.time_counter, ds[var + "_ice_weighted_mean", label="ICE")
        ax.set_ylabel(title)

    if ml_mid:
        integ_str = "ml_mid_horizontal_integ"
        render(axs[0], "votemper", integ_str, "Temperature")
        render(axs[1], "vosaline", integ_str, "Salinity")
        render(axs[3], "bn2", integ_str, r"N$^2$")
        render(axs[4], "bg_mod2", integ_str, r"$|\mathbf{\nabla}b|$")
    else:
        render(axs[0], "votemper", "domain_integ", "Temperature")
        render(axs[1], "vosaline", "domain_integ", "Salinity")
        render(axs[3], "N2", "horizontal_integ", r"N$^2$")
        render(axs[4], "bg_mod2", "domain_integ", r"$|\mathbf{\nabla}b|$")

    render(axs[2], "mld", "horizontal_integ", "MLD")
    render(axs[5], "windsp", "horizontal_integ", r"$U_{10}$")
    render(axs[6], "wfo", "horizontal_integ", r"$Q_{fw}$")
    render(axs[7], "taum", "horizontal_integ", r"$|\mathbf{\tau}_s|$")

    for ax in axs[:7]:
        ax.set_xticks([])

    axs[0].legend()

    if ml_mid: append = "_ml_mid" else append: = ""
    plt.savefig("glider_relevant_diags{}.pdf".format(append))

if __name__ == "__main__":

    plot_time_series("EXP10")
