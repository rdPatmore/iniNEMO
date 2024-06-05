import matplotlib.pyplot as plt
import xarray as xr
import config
import matplotlib

matplotlib.rcParams.update({"font.size": 8})

def plot_time_series(case):
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
    fig, axs = plt.subplots(5, figsize=(5.5,6))
    plt.subplots_adjust()

    # data source
    path = config.data_path() + case + "/TimeSeries/"

    # render Temperature
    T = xr.open_dataset(path + "votemper_domain_integ.nc")
    axs[0].plot(T.time_counter, T.votemper_oce_weighted_mean, label="Oce")
    axs[0].plot(T.time_counter, T.votemper_miz_weighted_mean, label="MIZ")
    axs[0].plot(T.time_counter, T.votemper_ice_weighted_mean, label="ICE")
    axs[0].set_ylabel("Temperature")

    # render Salinity
    S = xr.open_dataset(path + "vosaline_domain_integ.nc")
    axs[1].plot(S.time_counter, S.vosaline_oce_weighted_mean)
    axs[1].plot(S.time_counter, S.vosaline_miz_weighted_mean)
    axs[1].plot(S.time_counter, S.vosaline_ice_weighted_mean)
    axs[1].set_ylabel("Salinity")
    
    # render MLD
    mld = xr.open_dataset(path + "mld_horizontal_integ.nc")
    axs[2].plot(mld.time_counter, mld.mld_oce_weighted_mean)
    axs[2].plot(mld.time_counter, mld.mld_miz_weighted_mean)
    axs[2].plot(mld.time_counter, mld.mld_ice_weighted_mean)
    axs[2].set_ylabel("MLD")
    
    # render N2
    N2 = xr.open_dataset(path + "N2_mld_horizontal_integ.nc")
    axs[3].plot(N2.time_counter, N2.N2_mld_oce_weighted_mean)
    axs[3].plot(N2.time_counter, N2.N2_mld_miz_weighted_mean)
    axs[3].plot(N2.time_counter, N2.N2_mld_ice_weighted_mean)
    axs[3].set_ylabel("N$^2$")

    # render buoyancy gradinets
    bg = xr.open_dataset(path + "bg_mod2_domain_integ.nc")
    axs[4].plot(bg.time_counter, bg.bg_mod2_oce_weighted_mean)
    axs[4].plot(bg.time_counter, bg.bg_mod2_miz_weighted_mean)
    axs[4].plot(bg.time_counter, bg.bg_mod2_ice_weighted_mean)
    axs[4].set_ylabel(r"$|\mathbf{\nabla}b|$")

    for ax in axs[:4]:
        ax.set_xticks([])

    axs[0].legend()

    plt.savefig("glider_relevant_diags.pdf")

if __name__ == "__main__":

    plot_time_series("EXP10")
