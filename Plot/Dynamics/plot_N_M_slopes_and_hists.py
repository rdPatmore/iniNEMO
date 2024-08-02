import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
import matplotlib.colors as mc


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
    fig, axs = plt.subplots(3,figsize=(5.5,6.5))
    plt.subplots_adjust(hspace=0.5)

    # plot
    axs[0].bar(N2.bin_left, N2, align="edge", width=N2.bin_right - N2.bin_left)
    axs[1].bar(M2.bin_left, M2, align="edge", width=M2.bin_right - M2.bin_left)
    axs[2].bar(M2N2.bin_left, M2N2, align="edge", width=M2N2.bin_right - M2N2.bin_left)
    
    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('PDF')
    axs[0].set_xlabel(r'$N^2$')
    axs[1].set_xlabel(r'$M^2$')
    axs[2].set_xlabel(r'$M^2N^{-2}$')

    plt.show()
    #plt.savefig('N_M_slope.png', dpi=600)


def plot_N_M_histogram_2d():
    case = "EXP10"
    nc_preamble="SOCHIC_PATCH_3h_20121209_20130111_"
    # get N and M in mixed layer
    path = config.data_path() + case + "/BGHists/" + nc_preamble 
    N2 = xr.open_dataarray(path + "bn2_ml_mid_model_hist.nc")
    M2 = xr.open_dataarray(path + "bg_mod2_ml_mid_model_hist.nc")
    M2N2_2d = xr.open_dataarray(path + "bg_mod2_bn2_model_hist.nc")

    # initailise plot
    fig, axs = plt.subplots(1,figsize=(5.5,6.5))

    # plot
    axs.pcolor(M2N2_2d.x_bin_centers, M2N2_2d.y_bin_centers, M2N2_2d,
               norm=mc.LogNorm())
    
    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_xlabel('M2')
    axs.set_ylabel('N2')

    axs.set_aspect('equal')

    plt.show()

def plot_N_M_scatter():


plot_N_M_histogram_2d()
#plot_N_M_histogram()
