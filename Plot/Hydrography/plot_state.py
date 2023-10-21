import xarray as xr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import cmocean
import calendar

def plot_sochic_init():
    ini=0
    EXP='EXP11'
    prepend = '../Output/' + EXP
    if ini:
        ini = xr.open_dataset(prepend + '/output.init.nc')
        ini = ini.isel(x=slice(1,-1), y=slice(1,-1))
        #ini = ini.rename({'nav_lat': 'latt', 'nav_lon': 'lont'})
        lont = 'nav_lon'
        lonu = 'nav_lon'
        lonv = 'nav_lon'
        latt = 'nav_lat'
        latu = 'nav_lat'
        latv = 'nav_lat'
        time_step=0
    else:
        init = xr.open_dataset(prepend + '/SOCHIC_PATCH_20ts_20150101_'\
                                  '20150131_grid_T.nc')
        iniu = xr.open_dataset(prepend + '/SOCHIC_PATCH_20ts_20150101_'\
                                  '20150131_grid_U.nc')
        iniv = xr.open_dataset(prepend + '/SOCHIC_PATCH_20ts_20150101_'\
                                  '20150131_grid_V.nc')
        init = init.rename({'nav_lat': 'latt', 'nav_lon': 'lont',
                            'deptht':'nav_lev'})
        iniu = iniu.rename({'nav_lat': 'latu', 'nav_lon': 'lonu',
                            'depthu':'nav_lev'})
        iniv = iniv.rename({'nav_lat': 'latv', 'nav_lon': 'lonv',
                            'depthv':'nav_lev'})
        lont = 'lont'
        lonu = 'lont'
        lonv = 'lont'
        latt = 'latt'
        latu = 'latt'
        latv = 'latt'
        ini = xr.merge([init,iniu,iniv])
        ini = ini.isel(x=slice(1,-1), y=slice(1,-1))
        time_step=400
    
    # initialise plots
    fig = plt.figure(figsize=(6.5, 4.5), dpi=300)
    fig.suptitle('time: ' + str(ini.time_counter.values[time_step]))
    
    # initialise gridspec
    gs0 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
    gs1 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
    
    gs0.update(top=0.93, bottom=0.58, left=0.13)
    gs1.update(top=0.55, bottom=0.25, left=0.13)
    
    g = 9.81
    alpha = -3.2861e-5
    beta = 7.8358e-4
    axs0, axs1 = [], []
    for i in range(4):
        axs0.append(fig.add_subplot(gs0[i]))
        axs1.append(fig.add_subplot(gs1[i]))
    
    umin=ini.vozocrtx.min()
    umax=ini.vozocrtx.max()
    umax = np.abs(max(umin,umax))
    umin = - umax
    ulev = np.linspace(umin,umax,11)
    
    vmin=ini.vomecrty.min()
    vmax=ini.vomecrty.max()
    vmax = np.abs(max(vmin,vmax))
    vmin = - vmax
    vlev = np.linspace(vmin,vmax,11)
    
    smin = 33.5
    smax = 34.9
    slev = np.linspace(smin,smax,11)
    
    tmin = -2.0
    tmax = 2.0
    tlev = np.linspace(tmin,tmax,11)
    
    
    # plot surface
    ini_horiz = ini.isel(time_counter=time_step, nav_lev=0)
    p0 = axs0[0].contourf(ini_horiz[lonu], ini_horiz[latu], ini_horiz.vozocrtx,
                        levels=ulev, cmap=plt.cm.RdBu)
    p1 = axs0[1].contourf(ini_horiz[lonv], ini_horiz[latv], ini_horiz.vomecrty,
                        levels=vlev, cmap=plt.cm.RdBu)
    p2 = axs0[2].contourf(ini_horiz[lont], ini_horiz[latt], ini_horiz.votemper,
                          levels=tlev)
    p3 = axs0[3].contourf(ini_horiz[lont], ini_horiz[latt], ini_horiz.vosaline,
                        levels=slev)
    
    # plot vertical slice
    #ini = ini.assign_coords(x=np.arange(0,51), y=np.arange(0,100))
    ini_vert = ini.isel(time_counter=time_step, y=10)
    p0 = axs1[0].contourf(ini_vert[lonu], -ini_vert.nav_lev, ini_vert.vozocrtx,
                        levels=ulev, cmap=plt.cm.RdBu)
    p1 = axs1[1].contourf(ini_vert[lonv], -ini_vert.nav_lev, ini_vert.vomecrty,
                          levels=vlev, cmap=plt.cm.RdBu)
    p2 = axs1[2].contourf(ini_vert[lont], -ini_vert.nav_lev, ini_vert.votemper,
                          levels=tlev)
    p3 = axs1[3].contourf(ini_vert[lont], -ini_vert.nav_lev, ini_vert.vosaline,
                          levels=slev)
    
    # assign colour bar properties
    p = [p0,p1,p2,p3]
    labels = [r'u (m/s)',
              r'v (m/s)',
              r'Temperature ($^{\circ}$C)',
              r'Salinity (psu)']
    
    # add colour bars
    for i, ax in enumerate(axs1):
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
        cbar = fig.colorbar(p[i], cax=cbar_ax, orientation='horizontal')
        cbar.locator = ticker.MaxNLocator(nbins=3)
        cbar.update_ticks()
        cbar.ax.text(0.5, -4.5, labels[i], fontsize=8, rotation=0,
                     transform=cbar.ax.transAxes, va='bottom', ha='center')
    
    for ax in axs0:
        ax.set_xticks([])
    for ax in axs0[1:]:
        ax.set_yticks([])
    for ax in axs1:
        ax.set_xticklabels([-1.8,0,1.8])
    for ax in axs1[1:]:
        ax.set_yticks([])
    
    for ax in axs1:
        ax.set_xlabel('lon')
    axs0[0].set_ylabel('lat')
    axs1[0].set_ylabel('depth (m)')
    
    plt.savefig(EXP + '_state_t' + str(time_step) + '.png')

def plot_orca_state(pos='I'):
    '''
    plot all state variable of orca patch
    '''

    ds = xr.open_dataset('tmp/orca_I.nc')
    
    def plot(var, var_meta):
        print ('var: ', var)

        fig, axs = plt.subplots(2,6, figsize=(8,3))
        plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.15,
                            hspace=0.15, wspace=0.1)

        vmin = ds[var].min()
        vmax = ds[var].max()
        if var_meta['sym_bounds']:
            print (var_meta)
            vmax = max(np.abs(vmin), np.abs(vmax))
            vmin = -vmax
        if var == 'siconc':
            ds['siconc'] = ds.siconc.fillna(0.0)
        if var == 'sithic':
            ds['sithic'] = ds.sithic.fillna(0.0)

        for month in range(12):
            dsm = ds.isel(month=month)
            p = axs.flatten()[month].pcolor(dsm.nav_lon, dsm.nav_lat, dsm[var],
                                        vmin=vmin, vmax=vmax, shading='nearest',
                                        cmap=var_meta['cmap'])
            title = calendar.month_abbr[month+1]
            axs.flatten()[month].set_title(title)

        grid0 = axs[0,-1].get_position()
        grid1 = axs[1,-1].get_position()
        cbar_ax = fig.add_axes([0.89, grid1.y0, 0.02, grid0.y1 - grid1.y0])
        cbar = fig.colorbar(p, cax=cbar_ax)

        var_lookup = {'votemper': 'temperature',
                      'vosaline': 'salinity',
                      'vozocrtx': 'u',
                      'vomecrty': 'v',
                      'sowaflup': 'fresh water flux',
                      'soshfldo': 'shortwave radiation',
                      'sohefldo': 'heat flux',
                      'siconc': 'sea ice concentration',
                      'u_ice': 'sea zonal velocity',
                      'v_ice': 'sea meridional velocity',
                      'sitemp': 'sea ice temperature',
                      'sithic': 'sea ice thickness',
                      'snthic': 'snow thickness'}

        # colourbars
        var_human = var_lookup[var] 
        cbar.ax.text(5.0, 0.5, var_human, fontsize=8, rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='right')

        for ax in axs.flatten():
            ax.set_aspect('equal')
        for ax in axs[:,0]:
            ax.set_ylabel('latitude')
        for ax in axs[1,:]:
            ax.set_xlabel('longitude')
        for ax in axs[:1,:].flatten():
            ax.set_xticks([])
        for ax in axs[:,1:].flatten():
            ax.set_yticks([])

        plt.savefig('orca_state_' + pos + '_' + var + '.png', dpi=300)

    if pos == 'I':
        variables = {'siconc': {'cmap': cmocean.cm.ice_r, 'sym_bounds': False},
                     'sithic': {'cmap': cmocean.cm.ice_r, 'sym_bounds': False},
                     'snthic': {'cmap': cmocean.cm.ice_r, 'sym_bounds': False},
                     'sitemp': {'cmap': plt.cm.inferno  , 'sym_bounds': False},
                     'u_ice':  {'cmap': cmocean.cm.balance, 'sym_bounds': True},
                     'v_ice':  {'cmap': cmocean.cm.balance, 'sym_bounds': True},
                    }

    for var, var_meta in variables.items():
        plot(var, var_meta)

plot_orca_state(pos='I')
