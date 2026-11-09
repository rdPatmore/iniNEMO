import xarray as xr
import numpy  as np

def get_side(data, side, pos, res, exp, model, offset=0, rim_begin=1):
    print ('SIDE: ', side)
    mesh_path ='/work/n02/n02/ryapat30/nemo/nemoHEAD/cfgs/SOCHIC_ICE/' + exp 
    domain_cfg = xr.open_dataset(mesh_path + '/mesh_mask.nc').rename({
        'x':'X', 'y':'Y'}).squeeze('time_counter').reset_coords('time_counter')
    # the next three lines are inconsistent with make_coords
    # this is because data is missing rim0 information
    # need to make own version of SIREN for this component to include i=1
    domain_cfg = domain_cfg.isel(X=slice(1,-1),Y=slice(1,-1))
    vel_shift = 0
    br = 1 + offset - rim_begin # cell distance from model boundary
    if side == 'west':
        end = int(data.attrs['bdy_end']) - 1
        print (end)
        arrayX = data.isel(X=br,Y=slice(br,end-br)
                          ).reset_coords('X', drop=True)
        mesh_mask = domain_cfg.isel(X=br, Y=slice(br,end-br))
        #arrayX = data.isel(X=br,Y=slice(None,end)
        #                  ).reset_coords('X', drop=True)
        #mesh_mask = domain_cfg.isel(X=br, Y=slice(None,end))
        dim='Y'
        bdy_pos = int(arrayX.attrs['bdy_deb'] + br)

    if side == 'east':
        if pos == 'U':
            # this may need to be vel_shift = 1
            # vels exiting the domain are 1 in from tracer points
            vel_shift = 1 # jan 22 adjustment
            #vel_shift = 0 
        end = int(data.attrs['bdy_end']) - 1 # y north
        #arrayX = data.isel(X=vel_shift+br, Y=slice(None,end)
        #                   ).reset_coords('X', drop=True)  
        #mesh_mask = domain_cfg.isel(X=-1-vel_shift-br,
        #                            Y=slice(None,end))
        print (data)
        print ('end', end)
        print ('shift', vel_shift)
        print ('off', br)
        arrayX = data.isel(X=vel_shift+br, Y=slice(br,end-br)
                           ).reset_coords('X', drop=True)  
        mesh_mask = domain_cfg.isel(X=-1-vel_shift-br,
                                    Y=slice(br,end-br))
        bdy_pos = int(arrayX.attrs['bdy_ind'] + 1 - br - vel_shift)
        dim='Y'

    if side == 'south':
        end = int(data.attrs['bdy_end']) - 1
        #arrayX = data.isel(Y=br, X=slice(None,end)
        #                  ).reset_coords('Y', drop=True)
        #mesh_mask = domain_cfg.isel(Y=br, X=slice(None,end))
        arrayX = data.isel(Y=br, X=slice(br,end-br)
                          ).reset_coords('Y', drop=True)
        mesh_mask = domain_cfg.isel(Y=br, X=slice(br,end-br))
        dim='X'
        bdy_pos = int(arrayX.attrs['bdy_deb'] + br)

    if side == 'north':
        if pos == 'V':
            # this may need to be vel_shift = 1
            # vels exiting the domain are 1 in from tracer points
            vel_shift =1 
            #vel_shift =0 # jan 22 adjustment 
        end = int(data.attrs['bdy_end']) - 1
        #arrayX = data.isel(Y=vel_shift+br, X=slice(None,end)
        #                   ).reset_coords('Y', drop=True)
        #mesh_mask = domain_cfg.isel(Y=-1-vel_shift-br,
        #                   X=slice(None,end))
        arrayX = data.isel(Y=vel_shift+br, X=slice(br,end-br)
                           ).reset_coords('Y', drop=True)
        mesh_mask = domain_cfg.isel(Y=-1-vel_shift-br,
                           X=slice(br,end-br))
        bdy_pos = int(arrayX.attrs['bdy_ind'] + 1 - br - vel_shift)
        dim='X'
    if pos == 'T':
        arrayX['gdept'] = mesh_mask['gdept_0']
        arrayX['e3t'] = mesh_mask['e3t_0']
        if side in ['north', 'south']:
            nba = 'nbit'
            nbb = 'nbjt'
        if side in ['east', 'west']:
            nba = 'nbjt'
            nbb = 'nbit'
        if side in ['north']:
            arrayX = arrayX.sortby('X', ascending=False)
        if side in ['west']:
            arrayX = arrayX.sortby('Y', ascending=False)
        print (arrayX[dim].values+1)
        print (np.full(arrayX[dim].shape, bdy_pos))
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        #'nbrt': (['xbt'], np.full(arrayX[dim].shape,br+1)),
        # RDP 15/02/22 this is changing from start=1 to start =0 in order 
        # to capture 2 rims with orlanski rather than 1
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbt'], arrayX[dim].values + 1),
                         nbb: (['xbt'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbrt': (['xbt'], np.full(arrayX[dim].shape,br+1)),
                         'deptht':(['deptht'], arrayX.deptht.values),
                         'nav_lon':(['xbt'], arrayX.nav_lon.values),
                         'nav_lat':(['xbt'], arrayX.nav_lat.values),
                 'gdept':(['deptht','xbt'], arrayX.gdept.values),
                     'e3t':(['deptht','xbt'], arrayX.e3t.values),
              'sossheig':  (['time_counter','xbt'], arrayX.sossheig.values),
         'vosaline':  (['time_counter','deptht','xbt'], arrayX.vosaline.values),
         'votemper':  (['time_counter','deptht','xbt'], arrayX.votemper.values)}
                       ).expand_dims('yb')
        ds.deptht.attrs['long_name']='Vertical T levels'
        ds.deptht.attrs['units']='m'
        ds.vosaline.attrs['long_name']='Salinity'
        ds.vosaline.attrs['units']='psu'
        ds.votemper.attrs['long_name']='Temperature'
        ds.votemper.attrs['units']='degC'
        ds.nbrt.attrs['long_name']='bdy discrete distance'
        ds.nbrt.attrs['units']='unitless'
        ds.nbjt.attrs['long_name']='bdy j index'
        ds.nbjt.attrs['units']='unitless'
        ds.nbit.attrs['long_name']='bdy i index'
        ds.nbit.attrs['units']='unitless'
        ds = ds.transpose('time_counter','deptht','yb','xbt')
    if pos == 'U':
        arrayX['gdepu'] = mesh_mask['gdept_0']
        arrayX['e3u'] = mesh_mask['e3u_0']
        if side in ['north', 'south']:
            arrayX = arrayX.isel(X=slice(None,-1))
            nba = 'nbiu'
            nbb = 'nbju'
        if side in ['east', 'west']:
            nba = 'nbju'
            nbb = 'nbiu'
        if side in ['north']:
            arrayX = arrayX.isel(X=slice(None,None,-1))
        if side in ['west']:
            arrayX = arrayX.isel(Y=slice(None,None,-1))
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbu'], arrayX[dim].values + 1),
                         nbb: (['xbu'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbru': (['xbu'], np.full(arrayX[dim].shape,br+1)),
                         'depthu':(['depthu'], arrayX.depthu.values),
                         'nav_lat':(['xbu'], arrayX.nav_lat.values),
                         'nav_lon':(['xbu'], arrayX.nav_lon.values),
              'gdepu':(['depthu','xbu'], arrayX.gdepu.values),
                  'e3u':(['depthu','xbu'], arrayX.e3u.values),
        'vozocrtx':  (['time_counter','depthu','xbu'], arrayX.vozocrtx.values)}
                        ).expand_dims('yb')
        ds.depthu.attrs['long_name']='Vertical U levels'
        ds.depthu.attrs['units']='m'
        ds.vozocrtx.attrs['long_name']='Zonal velocity'
        ds.vozocrtx.attrs['units']='m/s'
        #ds.vozocrtx.attrs['missing_value']= 0
        ds.nbru.attrs['long_name']='bdy discrete distance'
        ds.nbru.attrs['units']='unitless'
        ds.nbju.attrs['long_name']='bdy j index'
        ds.nbju.attrs['units']='unitless'
        ds.nbiu.attrs['long_name']='bdy i index'
        ds.nbiu.attrs['units']='unitless'
        ds['vozocrtx'] = ds.vozocrtx.fillna(0.0)
        ds = ds.transpose('time_counter','depthu','yb','xbu')
    if pos == 'V':
        arrayX['gdepv'] = mesh_mask['gdept_0']
        arrayX['e3v'] = mesh_mask['e3v_0']
        if side in ['west', 'east']:
            arrayX = arrayX.isel(Y=slice(None,-1))
            nba = 'nbjv'
            nbb = 'nbiv'
        if side in ['north', 'south']:
            nba = 'nbiv'
            nbb = 'nbjv'
        if side in ['north']:
            arrayX = arrayX.isel(X=slice(None,None,-1))
        if side in ['west']:
            arrayX = arrayX.isel(Y=slice(None,None,-1))
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbv'], arrayX[dim].values + 1),
                         nbb: (['xbv'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbrv': (['xbv'], np.full(arrayX[dim].shape,br+1)),
                         'depthv':(['depthv'], arrayX.depthv.values),
                         'nav_lon':(['xbv'], arrayX.nav_lon.values),
                         'nav_lat':(['xbv'], arrayX.nav_lat.values),
                'gdepv':(['depthv','xbv'], arrayX.gdepv.values),
                    'e3v':(['depthv','xbv'], arrayX.e3v.values),
        'vomecrty':  (['time_counter','depthv','xbv'], arrayX.vomecrty.values)}
                       ).expand_dims('yb')
        ds.depthv.attrs['long_name']='Vertical V levels'
        ds.depthv.attrs['units']='m'
        ds.vomecrty.attrs['long_name']='Meridional velocity'
        ds.vomecrty.attrs['units']='m/s'
        #ds.vomecrty.attrs['missing_value']= 0
        ds.nbrv.attrs['long_name']='bdy discrete distance'
        ds.nbrv.attrs['units']='unitless'
        ds.nbjv.attrs['long_name']='bdy j index'
        ds.nbjv.attrs['units']='unitless'
        ds.nbiv.attrs['long_name']='bdy i index'
        ds.nbiv.attrs['units']='unitless'
        ds['vomecrty'] = ds.vomecrty.fillna(0.0)
        ds = ds.transpose('time_counter','depthv','yb','xbv')
    if pos == 'I':
        if side in ['north', 'south']:
            nba = 'nbit'
            nbb = 'nbjt'
        if side in ['east', 'west']:
            nba = 'nbjt'
            nbb = 'nbit'
        if side in ['north']:
            arrayX = arrayX.sortby('X', ascending=False)
        if side in ['west']:
            arrayX = arrayX.sortby('Y', ascending=False)
        arrayX = arrayX.swap_dims({'T':'time_counter'})
        ds = xr.Dataset({'time_counter': arrayX.time_counter,
                         nba: (['xbt'], arrayX[dim].values + 1),
                         nbb: (['xbt'], np.full(arrayX[dim].shape, bdy_pos)),
                         'nbrt': (['xbt'], np.full(arrayX[dim].shape,br+1)),
                         'nav_lon':(['xbt'], arrayX.nav_lon.values),
                         'nav_lat':(['xbt'], arrayX.nav_lat.values),
         'siconc':  (['time_counter','xbt'], arrayX.siconc.values),
         'sithic':  (['time_counter','xbt'], arrayX.sithic.values),
         'snthic':  (['time_counter','xbt'], arrayX.snthic.values),
         'sitemp':  (['time_counter','xbt'], arrayX.sitemp.values),
         'sisalt':  (['time_counter','xbt'], arrayX.sisalt.values)}
                       ).expand_dims('yb')
        ds.siconc.attrs['long_name']='Sea Ice Concentration'
        ds.sithic.attrs['long_name']='Sea Ice Thickness'
        ds.snthic.attrs['long_name']='Snow Thickness'
        ds.sitemp.attrs['long_name']='Sea Ice Temperature'
        ds.sisalt.attrs['long_name']='Sea Ice Salinity'
        ds.snthic.attrs['long_name']='Snow Thickness'
        ds.siconc.attrs['units']='unitless'
        ds.sithic.attrs['units']='m'
        ds.snthic.attrs['units']='m'
        ds.sitemp.attrs['units']='decC'
        ds.sisalt.attrs['units']='psu'
        ds.nbrt.attrs['long_name']='bdy discrete distance'
        ds.nbrt.attrs['units']='unitless'
        ds.nbjt.attrs['long_name']='bdy j index'
        ds.nbjt.attrs['units']='unitless'
        ds.nbit.attrs['long_name']='bdy i index'
        ds.nbit.attrs['units']='unitless'
        ds = ds.transpose('time_counter','yb','xbt')
    ds.nav_lat.attrs['units']='degrees_north'
    ds.nav_lon.attrs['units']='degrees_east'
    ds.attrs['history'] = 'Created using RDPs NEMO config on SCIHUB'
    return ds

def single_bound(data, mesh_mask, side, pos, res, width=1):
    if width == 1:
        ds = get_side(data, side, pos, res, offset=0)
    else:
        segments = []
        for i in range(0,width):
            segments.append(get_side(data, side, pos, res, offset=i))
        if pos in ['U','V','T']:
            ds = xr.concat(segments, dim=('xb' + pos).lower())
        elif pos is 'I':
            ds = xr.concat(segments, dim='xbt')
    return ds

def get_ring(pos, width, date, res, exp, model):
    segments = []
    for ring in range(0,width):
        print ('RING', ring)
        for side in ['east', 'north', 'west', 'south']:
            path = '../DataOut/' + model + '/ORCA' + res
            append = side + '_' + date + '_masked.nc'
            if pos == 'T':
                data_pathT = path + '/Masked/bdy_T_' + append
                data = xr.open_dataset(data_pathT, decode_times=False)
            if pos == 'U':
                data_pathU = path + '/Masked/bdy_U_' + append
                data = xr.open_dataset(data_pathU, decode_times=False)
            if pos == 'V':
                data_pathV = path + '/Masked/bdy_V_' + append
                data = xr.open_dataset(data_pathV, decode_times=False)
            if pos == 'I':
                data_pathI = path + '/Masked/bdy_I_' + append
                data = xr.open_dataset(data_pathI, decode_times=False)
            #orcaT_path = '../processORCA12/DataIn/ORCA0083-N06_20150105d05T.nc'
            #orca_time = xr.open_dataset(orcaT_path).time_counter
            #print (data)
            #data['time_counter'] = orca_time
            #segments.append(get_side(data, side, pos, res, offset=ring))
            #segments.append(get_side(data, side, pos, res, offset=ring-1)) # jan 22 adjustment
            segments.append(get_side(data, side, pos, res, exp, model,
                            offset=ring, rim_begin=1)) # feb 22 adjustment
    if pos in ['U','V','T']:
        pos = pos
    elif pos is 'I':
        pos = 't'
    return xr.concat(segments, dim=('xb' + pos).lower())

def full_bounds(width, date='y2015m01', res='12', exp='EXP45', model='SATLAN'):
    #mesh_mask = xr.open_dataset('mesh_mask.nc')#.isel(
    #              x=slice(None,10), y=slice(1,99)).rename(
    #            {'x':'X','y':'Y'}).rename({'time_counter':'tc'})
    dsI = get_ring('I', date=date, width=width, res=res, exp=exp, model=model)
    dsT = get_ring('T', date=date, width=width, res=res, exp=exp, model=model)
    dsU = get_ring('U', date=date, width=width, res=res, exp=exp, model=model)
    dsV = get_ring('V', date=date, width=width, res=res, exp=exp, model=model)
    path = '../DataOut/' + model + '/ORCA' + res
    dsT.to_netcdf(path + '/ModelReady/bdy_T_ring_' + date + '.nc',
                  unlimited_dims='time_counter')
    dsU.to_netcdf(path +  '/ModelReady/bdy_U_ring_' + date + '.nc',
                  unlimited_dims='time_counter')
    dsV.to_netcdf(path +'/ModelReady/bdy_V_ring_' + date + '.nc',
                   unlimited_dims='time_counter')
    dsI.to_netcdf(path + '/ModelReady/bdy_I_ring_' + date + '.nc',
                   unlimited_dims='time_counter')

def all_pos_one_side(side, width=1):
    data_pathT = '../Masks/BdyData/bdy_T_west_masked.nc'
    dataT = xr.open_dataset(data_pathT, decode_times=False)
    data_pathU = '../Masks/BdyData/bdy_U_west_masked.nc'
    dataU = xr.open_dataset(data_pathU, decode_times=False)
    data_pathV = '../Masks/BdyData/bdy_V_west_masked.nc'
    dataV = xr.open_dataset(data_pathV, decode_times=False)
    orcaT_path = '../processORCA12/DataIn/ORCA0083-N06_20150105d05T.nc'
    orca_time = xr.open_dataset(orcaT_path).time_counter
    mesh_mask = xr.open_dataset('mesh_mask.nc').isel(
                  x=slice(None,10), y=slice(1,99)).rename(
                {'x':'X','y':'Y'}).rename({'time_counter':'tc'})
    dataT['time_counter'] = orca_time
    dataU['time_counter'] = orca_time
    dataV['time_counter'] = orca_time
    dataT['gdept'] = mesh_mask['gdept_0']
    dataU['gdepu'] = mesh_mask['gdept_0']
    dataV['gdepv'] = mesh_mask['gdept_0']
    dataT['e3t'] = mesh_mask['e3t_0']
    dataU['e3u'] = mesh_mask['e3u_0']
    dataV['e3v'] = mesh_mask['e3v_0']
    dsT = single_bound(dataT, side, 'T', width=width)
    dsU = single_bound(dataU, side, 'U', width=width)
    dsV = single_bound(dataV, side, 'V', width=width)
    print (dsT)
    print (dsU)
    print (dsV)
    dsT.to_netcdf('DataOut/bdy_T_west_masked.nc', unlimited_dims='time_counter')
    dsU.to_netcdf('DataOut/bdy_U_west_masked.nc', unlimited_dims='time_counter')
    dsV.to_netcdf('DataOut/bdy_V_west_masked.nc', unlimited_dims='time_counter')

   
#bdy_coords = get_ring(coords, 0)
#if rings > 1:
#    for ring in range(1, rings):
#        print ('ring: ', ring)
#        bdy_coords = xr.concat([bdy_coords, get_ring()], dim='xbT')
#
#bdy_coords.to_netcdf('coordinates_bdy.nc')

model='SATLAN'
exp='EXP50'

full_bounds(32, date='y2012', res='96', exp=exp, model=model)
full_bounds(32, date='y2013', res='96', exp=exp, model=model)
full_bounds(32, date='y2014', res='96', exp=exp, model=model)
