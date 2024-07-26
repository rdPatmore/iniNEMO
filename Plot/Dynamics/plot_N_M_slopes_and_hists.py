import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np


def plot_N_M_histogram():
    ''' '''

    case = "EXP10"
    nc_preamble="SOCHIC_PATCH_3h_20121209_20130111_"
    # get N and M in mixed layer
    path = config.data_path() + case + "/BGHists/" + nc_preamble 
    N2 = xr.open_dataarray(path + "bn2_ml_mid_model_hist.nc")
    M2 = xr.open_dataarray(path + "bg_mod2_ml_mid_model_hist.nc")
    M2N2 = xr.open_dataarray(path + "M2_over_N2_ml_mid_model_hist.nc")
                           

    # initailise plot
    fig, axs = plt.subplots(3)

    # plot
    axs[0].bar(N2.bin_left, N2, align="edge", width=N2.bin_right - N2.bin_left)
    axs[1].bar(M2.bin_left, M2, align="edge", width=M2.bin_right - M2.bin_left)
    axs[2].bar(M2N2.bin_left, M2N2, align="edge", width=M2N2.bin_right - M2N2.bin_left)
    
    for ax in axs:
        ax.set_yscale('log')
    #axs[2].hist2d(N2, M2)

    plt.show()
plot_N_M_histogram()


#def plot_slope_historgram():
#
#def plot_slope_map():


