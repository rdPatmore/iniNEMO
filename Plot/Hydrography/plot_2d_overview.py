import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import config
import cmocean
import iniNEMO.Process.Common.spatial_integrals_and_masking as sim

matplotlib.rcParams.update({'font.size': 8})

def three_panel():
    # load data 
    file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
    raw_preamble = config.data_path() + 'EXP10/RawOutput/' +  file_id
    proc_preamble = config.data_path() + 'EXP10/ProcessedVars/' +  file_id
    kwargs = {'chunks':'auto' ,'decode_cf':True} 
    mld = xr.open_dataset(raw_preamble + 'grid_T.nc', **kwargs).mldr10_3
    bg_mod2 = xr.open_dataset(
                proc_preamble + 'bg_mod2.nc', **kwargs).bg_mod2**0.5
    ice = xr.open_dataset(raw_preamble + 'icemod.nc', **kwargs).siconc
    
    # select time
    tslice = '2012-12-24 12:00:00'
    
    depth = 10
    sel_dict = dict(time_counter=tslice, deptht=depth)
    mld  = mld.sel(time_counter=tslice, method='nearest')
    ice  = ice.sel(time_counter=tslice, method='nearest')
    bg_mod2  = bg_mod2.sel(sel_dict, method='nearest')
    
    # cut boundaries
    bounds = dict(x=slice(45,-45), y=slice(45,-45))
    mld = mld.isel(bounds)
    ice = ice.isel(bounds)
    bg_mod2 = bg_mod2.isel(bounds)
    
    # plot
    fig, axs = plt.subplots(1,3, figsize=(6.5,3.0))
    plt.subplots_adjust(bottom=0.3, top=0.99, right=0.98, left=0.1, hspace=0.05)
    p0 = axs[0].pcolor(mld.nav_lon, mld.nav_lat, mld, vmin=0, vmax=120,
                       shading='nearest')
    p1 = axs[1].pcolor(bg_mod2.nav_lon, bg_mod2.nav_lat, bg_mod2,
                       vmin=0, vmax=4e-7,
                       shading='nearest', cmap=plt.cm.binary)
    p2 = axs[2].pcolor(ice.nav_lon, ice.nav_lat, ice, cmap=plt.cm.RdBu,
                       vmin=-1, vmax=1, shading='nearest')
    p = [p0,p1,p2]
    txt = ['Mixed Layer Depth (m)',
            r'M$^2$ (s$^{-2}$)',
            'Sea Ice Concentration (-)']
    
    for i, ax in enumerate(axs):
        # aspect ratio
        ax.set_aspect('equal')
    
        # colorbars
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.15, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -5.0, txt[i], fontsize=8,
                     rotation=0, transform=cbar.ax.transAxes,
                     va='center', ha='center')
    
    # set axis labels
    axs[0].set_ylabel(r'Latitude ($^{\circ}$N)')
    for ax in axs:
        ax.set_xlabel(r'Longitude ($^{\circ}$E)')
    for ax in axs[1:]:
        ax.set_yticklabels([])
    
    # set axes aspect
    for ax in axs:
        ax.set_aspect('equal')
    
    # save
    plt.savefig('mld_bg_Ro_' + tslice.split(' ')[0] +  '.png', dpi=300)

def four_panel():
    # load data 
    file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
    raw_preamble = config.data_path() + 'EXP10/RawOutput/' +  file_id
    proc_preamble = config.data_path() + 'EXP10/ProcessedVars/' +  file_id
    kwargs = {'chunks':'auto' ,'decode_cf':True} 
    mld = xr.open_dataset(raw_preamble + 'grid_T.nc', **kwargs).mldr10_3
    T = xr.open_dataset(raw_preamble + 'grid_T.nc', **kwargs).votemper
    bg_mod2 = xr.open_dataset(
                proc_preamble + 'bg_mod2.nc', **kwargs).bg_mod2**0.5
    ice = xr.open_dataset(raw_preamble + 'icemod.nc', **kwargs).siconc
    
    # select time
    tslice = '2012-12-24 12:00:00'
    
    depth = 10
    sel_dict = dict(time_counter=tslice, deptht=depth)
    mld  = mld.sel(time_counter=tslice, method='nearest')
    ice  = ice.sel(time_counter=tslice, method='nearest')
    bg_mod2  = bg_mod2.sel(sel_dict, method='nearest')
    T = T.sel(sel_dict, method='nearest')
    
    # cut boundaries
    bounds = dict(x=slice(10,-10), y=slice(10,-10))
    mld = mld.isel(bounds)
    ice = ice.isel(bounds)
    bg_mod2 = bg_mod2.isel(bounds)
    T = T.isel(bounds)
    
    # plot
    fig, axs = plt.subplots(1,4, figsize=(8.5,3.0))
    plt.subplots_adjust(bottom=0.3, top=0.99, right=0.98, left=0.1, hspace=0.05)
    p0 = axs[0].pcolor(T.nav_lon, T.nav_lat, T, cmap=cmocean.cm.thermal,
                       vmin=-1.5, vmax=0.5, shading='nearest')
    p1 = axs[1].pcolor(mld.nav_lon, mld.nav_lat, mld, vmin=0, vmax=120,
                       shading='nearest')
    p2 = axs[2].pcolor(bg_mod2.nav_lon, bg_mod2.nav_lat, bg_mod2,
                       vmin=0, vmax=4e-7,
                       shading='nearest', cmap=plt.cm.binary)
    p3 = axs[3].pcolor(ice.nav_lon, ice.nav_lat, ice, cmap=cmocean.cm.ice,
                       vmin=0, vmax=1, shading='nearest')
    p = [p0,p1,p2,p3]
    txt = [r'Potential Temperature ($^{\circ}$C)',
           'Mixed Layer Depth (m)',
            r'M$^2$ (s$^{-2}$)',
            'Sea Ice Concentration (-)']
    
    for i, ax in enumerate(axs):
        # aspect ratio
        ax.set_aspect('equal')
    
        # colorbars
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.15, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -5.0, txt[i], fontsize=8,
                     rotation=0, transform=cbar.ax.transAxes,
                     va='center', ha='center')
    
    # set axis labels
    axs[0].set_ylabel(r'Latitude ($^{\circ}$N)')
    for ax in axs:
        ax.set_xlabel(r'Longitude ($^{\circ}$E)')
    for ax in axs[1:]:
        ax.set_yticklabels([])
    
    # set axes aspect
    for ax in axs:
        ax.set_aspect('equal')
    
    # save
    plt.savefig('T_mld_bg_ice_' + tslice.split(' ')[0] +  '.png', dpi=300)

def ice_mask():
    # load data 
    file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
    raw_preamble = config.data_path() + 'EXP10/RawOutput/' +  file_id
    proc_preamble = config.data_path() + 'EXP10/ProcessedVars/' +  file_id
    kwargs = {'chunks':'auto' ,'decode_cf':True} 
    mld = xr.open_dataset(raw_preamble + 'grid_T.nc', **kwargs).mldr10_3
    T = xr.open_dataset(raw_preamble + 'grid_T.nc', **kwargs).votemper
    bg_mod2 = xr.open_dataset(
                proc_preamble + 'bg_mod2.nc', **kwargs).bg_mod2**0.5
    ice = xr.open_dataset(raw_preamble + 'icemod.nc', **kwargs).siconc
    

    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    im = sim.integrals_and_masks('EXP10', file_id, ice, 'siconc')
    im.get_domain_vars_and_cut_rims()
    ice_miz, ice_ice, ice_oce = im.mask_by_ice_oce_zones(threshold=0.2)

    # select time
    tslice = '2012-12-09 12:00:00'
    sel_dict = dict(time_counter=tslice)
    ice_miz  = ice_miz.sel(time_counter=tslice, method='nearest')
    ice_ice  = ice_ice.sel(time_counter=tslice, method='nearest')
    ice_oce  = ice_oce.sel(time_counter=tslice, method='nearest')

    # plot
    fig, axs = plt.subplots(1,3, figsize=(6.5,3.0))
    plt.subplots_adjust(bottom=0.3, top=0.99, right=0.98, left=0.1, hspace=0.05)
    p0 = axs[0].pcolor(ice_miz.nav_lon, ice_miz.nav_lat, ice_miz,
                       cmap=cmocean.cm.ice,
                       vmin=0, vmax=1, shading='nearest')
    p1 = axs[1].pcolor(ice_ice.nav_lon, ice_ice.nav_lat, ice_ice,
                       cmap=cmocean.cm.ice,
                       vmin=0, vmax=1, shading='nearest')
    p2 = axs[2].pcolor(ice_oce.nav_lon, ice_oce.nav_lat, ice_oce,
                       cmap=cmocean.cm.ice,
                       vmin=0, vmax=1, shading='nearest')
    p = [p0,p1,p2]
    txt = ['MIZ Sea Ice Conc. (-)',
           'ICE Sea Ice Conc. (-)',
           'OCE Sea Ice Conc. (-)']
    
    for i, ax in enumerate(axs):
        # aspect ratio
        ax.set_aspect('equal')
    
        # colorbars
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.15, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -5.0, txt[i], fontsize=8,
                     rotation=0, transform=cbar.ax.transAxes,
                     va='center', ha='center')
    
    # set axis labels
    axs[0].set_ylabel(r'Latitude ($^{\circ}$N)')
    for ax in axs:
        ax.set_xlabel(r'Longitude ($^{\circ}$E)')
    for ax in axs[1:]:
        ax.set_yticklabels([])
    
    # save
    plt.savefig('ice_mask_' + tslice.split(' ')[0] +  '.png', dpi=1200)
ice_mask()
