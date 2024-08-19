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
    model = "SOCHIC_PATCH_3h_"
    #quad = "_lower_right"
    quad = ""
    time_str = "20121209_20130111"
    #time_str = "20121223"
    space_time_str = time_str + quad
    nc_preamble = model + space_time_str
    # get N and M in mixed layer
    path = config.data_path() + case + "/BGHists/" + nc_preamble 
    N2 = xr.open_dataset(path + "_bn2_ml_mid_ice_oce_miz_model_hist.nc")
    M2 = xr.open_dataset(path + "_bg_mod2_ml_mid_ice_oce_miz_model_hist.nc")
    M2N2 = xr.open_dataset(
                      path + "_M2_over_N2_ml_mid_ice_oce_miz_model_hist.nc")

    # initailise plot
    fig, axs = plt.subplots(3,figsize=(5.5,6.5))
    plt.subplots_adjust(hspace=0.5)

    def render(N2, M2, M2N2, partition, quad, c='r'):
        N2 = N2['hist_bn2_ml_mid_' + partition + quad]
        M2 = M2['hist_bg_mod2_ml_mid_' + partition + quad]
        M2N2 = M2N2['hist_M2_over_N2_ml_mid_' + partition + quad]
        #axs[0].bar(N2.bin_left, N2, align="edge",
        #           width=N2.bin_right - N2.bin_left, color=c, alpha=0.2)
        #axs[1].bar(M2.bin_left, M2, align="edge",
        #           width=M2.bin_right - M2.bin_left, color=c, alpha=0.2)
        #axs[2].bar(M2N2.bin_left, M2N2, align="edge",
        #           width=M2N2.bin_right - M2N2.bin_left, color=c, alpha=0.2)
        axs[0].step(N2.bin_centers, N2, where="mid",
                   color=c, label=partition)
        axs[1].step(M2.bin_centers, M2, where="mid",
                   color=c, label=partition)
        axs[2].step(M2N2.bin_centers, M2N2, where="mid",
                   color=c, label=partition)

    # plot
    render(N2, M2, M2N2, 'ice', quad, c='r')
    render(N2, M2, M2N2, 'oce', quad, c='g')
    render(N2, M2, M2N2, 'miz', quad, c='b')

    axs[0].legend(loc='upper left') 
    
    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('Count')
        ax.set_ylim(0.5,5e7)
    axs[0].set_xlabel(r'$N^2$')
    axs[1].set_xlabel(r'$M^2$')
    axs[2].set_xlabel(r'$M^2N^{-2}$')

    plt.suptitle(time_str)
    
    #plt.show()
    plt.savefig('N_M_slope_{}.png'.format(space_time_str), dpi=600)

def plot_N_M_histogram_2d():
    case = "EXP10"
    #nc_preamble="SOCHIC_PATCH_3h_20121209_20130111_"
    model = "SOCHIC_PATCH_3h_"
    quad = ""
    #quad = "_lower_right"
    time_str = "20121209_20130111"
    space_time_str = time_str + quad
    nc_preamble = model + space_time_str
    # get N and M in mixed layer
    path = config.data_path() + case + "/BGHists/" + nc_preamble 
    M2N2_2d = xr.open_dataset(
             path + "_bg_mod2_bn2_ml_mid_ice_oce_miz_model_hist.nc")

    # initailise plot
    fig, axs = plt.subplots(3 ,figsize=(3.2,5.5))
    plt.subplots_adjust(right=0.80, top=0.93)

    def render(ax, partition, quad):
        p = ax.pcolor(M2N2_2d.x_bin_centers, M2N2_2d.y_bin_centers,
                       M2N2_2d['hist_bg_mod2_ml_mid_' + partition + quad + 
                               '_bn2_ml_mid_' + partition + quad].T,
                       norm=mc.LogNorm(1,2e6))

        # add colour bar
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.82, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(7.0, 0.5, 'Count',
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='right')

        ax.plot([M2N2_2d.y_bin_centers.min(),
                 M2N2_2d.y_bin_centers.max()],
                [M2N2_2d.y_bin_centers.min(),
                 M2N2_2d.y_bin_centers.max()])
        ax.set_xlim(M2N2_2d.x_bin_centers.min(),
                    M2N2_2d.x_bin_centers.max())
        ax.set_ylim(M2N2_2d.y_bin_centers.min(),
                    M2N2_2d.y_bin_centers.max())


    # plot
    render(axs[0], 'ice', quad)
    render(axs[1], 'oce', quad)
    render(axs[2], 'miz', quad)
    
    labels = ['Ice','Oce','MIZ']
    for i, ax in enumerate(axs):
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_ylabel('N2')

        ax.set_aspect('equal')

        ax.text(0.1,0.1, labels[i], ha='left',va='bottom',
                transform=ax.transAxes)

    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[-1].set_xlabel('M2')

    plt.suptitle(time_str)

    #plt.savefig('M2_N2_2d_histogram.png', dpi=600)
    plt.savefig('M2_N2_2d_histogram_{}.png'.format(space_time_str), dpi=600)

def cut_time(var, dates):
      ''' reduce time period to bounds '''
      var = var.sel(time_counter=slice(dates[0],dates[1]))
      return var

def plot_N_M_scatter():
    ''' scater plot of N2 versus M2 '''

    case = "EXP10"
    nc_preamble="SOCHIC_PATCH_3h_20121209_20130331_"
    path = config.data_path() + case + "/ProcessedVars/" + nc_preamble 
    M2 = xr.open_dataset(path + 'bg_mod2_ml_mid_ice_oce_miz.nc',
                           chunks='auto')
    N2 = xr.open_dataset(path + 'bn2_ml_mid_ice_oce_miz.nc', chunks='auto')
    #N2 = N2.isel(x=slice(2,-2),y=slice(2,-2))

    M2 = cut_time(M2, ['20121209','20130111'])
    N2 = cut_time(N2, ['20121209','20130111'])

    N2_stack = N2.stack(z=('time_counter','x','y'))
    M2_stack = M2.stack(z=('time_counter','x','y'))

    # initailise plot
    fig, axs = plt.subplots(1,3,figsize=(6.5,6.5))

    def render(ax, var, c='r'):
        ax.scatter(M2_stack['bg_mod2_ml_mid_' + var], 
                    N2_stack['bn2_ml_mid_' + var], s=0.1, c=c)

    render(axs[0], 'ice', c='r')
    render(axs[1], 'oce', c='g')
    render(axs[2], 'miz', c='b')

    for ax in axs:
        # set log10 axes scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # set axes labels
        ax.set_xlabel('M2')
        ax.set_ylabel('N2')

        ax.set_aspect('equal')

        ax.set_xlim([1e-26,1e-11])

    plt.savefig('M2_N2_scatter.png', dpi=600)

#plot_N_M_scatter()
plot_N_M_histogram_2d()
#plot_N_M_histogram()
