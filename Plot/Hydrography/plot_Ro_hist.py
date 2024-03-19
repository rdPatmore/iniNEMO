import xarray as xr
import matplotlib.pyplot as plt
import config
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

def render_rossby_number_hist(ax):
    '''
    render histogram of the rossby number at the surface
    '''

    # load
    file_id = '/BGHists/SOCHIC_PATCH_3h_20121209_20130331_'
    hist_ro = xr.open_dataset(config.data_path() + 'EXP10' +
                              file_id + 'Ro_model_hist.nc').load()

    print (hist_ro)
    # get step boundaries
    stair_edges = np.unique(np.concatenate((hist_ro.bin_left.values, \
                                           hist_ro.bin_right.values)))

    # plot
    c1 = '#f18b00'
    ax.stairs(hist_ro.hist_ro, stair_edges, orientation='horizontal',
              color=c1, lw=0.8)

    # x params
    ax.xaxis.get_offset_text().set_visible(False)
    ax.set_xlabel(r'PDF ($\times 10 ^{8}$)')

    # x params
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_ylabel(r'$\zeta/f$ (-)')
    ax.set_ylim(stair_edges[0],stair_edges[-1])


def plot_rossby_number():
    '''
    plot probability density function of Ro
    '''

    # initialise figure
    fig, ax = plt.subplots()

    render_rossby_number_hist(ax)
    plt.savefig('rossby_number_pdf_ln.png')
plot_rossby_number()
