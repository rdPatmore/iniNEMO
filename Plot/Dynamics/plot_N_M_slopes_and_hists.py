import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
import matplotlib.colors as mc
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

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

    #plt.show()
    plt.savefig('N_M_slope.png', dpi=600)


def plot_N_M_histogram_2d():
    case = "EXP10"
    nc_preamble="SOCHIC_PATCH_3h_20121209_20130111_"
    # get N and M in mixed layer
    path = config.data_path() + case + "/BGHists/" + nc_preamble 
    N2 = xr.open_dataarray(path + "bn2_ml_mid_model_hist.nc")
    M2 = xr.open_dataarray(path + "bg_mod2_ml_mid_model_hist.nc")
    M2N2_2d = xr.open_dataarray(path + "bg_mod2_bn2_model_hist.nc")

    # initailise plot
    fig, axs = plt.subplots(1,figsize=(5.5,4.5))
    plt.subplots_adjust(right=0.80)

    # plot
    p = axs.pcolor(M2N2_2d.x_bin_centers, M2N2_2d.y_bin_centers, M2N2_2d.T,
               norm=mc.LogNorm())
    
    axs.set_xscale('log')
    axs.set_yscale('log')

    axs.set_xlabel('M2')
    axs.set_ylabel('N2')

    axs.set_aspect('equal')

    # add colour bar
    pos = axs.get_position()
    cbar_ax = fig.add_axes([0.82, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(6.0, 0.5, 'PDF',
               rotation=90, transform=cbar.ax.transAxes,
               va='center', ha='right')

    plt.savefig('M2_N2_2d_histogram.png', dpi=600)

def cut_time(var, dates):
      ''' reduce time period to bounds '''
      var = var.sel(time_counter=slice(dates[0],dates[1]))
      return var

def plot_N_M_scatter():
    ''' scater plot of N2 versus M2 '''

    case = "EXP10"
    nc_preamble="SOCHIC_PATCH_3h_20121209_20130331_"
    path = config.data_path() + case + "/ProcessedVars/" + nc_preamble 
    M2 = xr.open_dataarray(path + 'bg_mod2_ml_mid_ice_oce_miz.nc',
                           chunks='auto')
    N2 = xr.open_dataarray(path + 'bn2_ml_mid_ice_oce_miz.nc', chunks='auto')
    print (N2)
    print (M2)
    N2 = N2.isel(x=slice(2,-2),y=slice(2,-2))

    M2 = cut_time(M2, ['20121209','20130111'])
    N2 = cut_time(N2, ['20121209','20130111'])

    N2_stack = N2.stack(z=('time_counter','x','y'))
    M2_stack = M2.stack(z=('time_counter','x','y'))

    # initailise plot
    fig, axs = plt.subplots(1,figsize=(5.5,6.5))

    axs.scatter(M2_stack, N2_stack, s=0.1)

    # set log10 axes scales
    axs.set_xscale('log')
    axs.set_yscale('log')

    # set axes labels
    axs.set_xlabel('M2')
    axs.set_ylabel('N2')

    axs.set_aspect('equal')

    axs.set_xlim([1e-26,M2.max()])

    plt.savefig('M2_N2_scatter.png', dpi=600)

#plot_N_M_scatter()
plot_N_M_histogram_2d()
#plot_N_M_histogram()
