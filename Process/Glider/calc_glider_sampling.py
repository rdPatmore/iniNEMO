import iniNEMO.Process.Common.model_object as model_object
import numpy as np
import dask
import config
import xarray as xr

def glider_sampling(case, remove=False, append='', interp_dist=1000,
                    transects=False, south_limit=None, north_limit=None,
                    rotate=False, rotation=np.pi/2):

    m = model_object.model(case)
    m.interp_dist=interp_dist
    #m.transects=transects
    m.save_append = 'interp_' + str(interp_dist) + append
    if remove:
        m.save_append = m.save_append + '_' + remove
    if transects:
        m.save_append = m.save_append + '_pre_transect'
    m.load_gridT_and_giddy()

    # reductions of nemo domain
    m.south_limit = south_limit
    m.north_limit = north_limit

    #m.save_area_mean_all()
    #m.save_area_std_all()
    #m.save_month()
    #m.get_conservative_temperature(save=True)
    #sample_dist=5000
    #m.prep_interp_to_raw_obs(resample_path=True, sample_dist=sample_dist)
    m.prep_interp_to_raw_obs(rotate=rotate, rotation=rotation)
    if transects:
        #m.get_transects(shrink=100)
        print (m.giddy_raw)
        m.giddy_raw = get_transects(m.giddy_raw, 
                                    method='from interp_1000',
                                    shrink=100)
    if remove:
        m.prep_remove_dives(remove=remove)
    for ind in range(0,100):
        m.ind = ind
        print ('ind: ', ind)
        # calculate perfect gradient crossings
        m.interp_to_raw_obs_path(random_offset=True, load_offset=True)
        print ('done part 1')
        m.interp_raw_obs_path_to_uniform_grid(ind=ind)
        print ('done part 2')
    #inds = np.arange(100)
    #m.ds['grid_T'] = m.ds['grid_T'].expand_dims(ind=inds)
    #futures = client.map(process_all, inds, **dict(m=m))
    #client.gather(futures)
    #xr.apply_ufunc(process_all, inds, dask="parallelized")
    print (' ')
    print ('successfully ended')
    print (' ')

def glider_sample_parallel_straight_line_paths():
    ''' get 100 samples of model with 2 parallel transects '''

    # offset in parallel transects
    lon_shift = 1/12.

    # initialise object
    m = model_object.model('EXP10')

    # get data and set file naming
    m.interp_dist=1000
    m.save_append = 'interp_' + str(m.interp_dist) + '_parallel_transects'
    m.load_gridT_and_giddy(g_fn='artificial_straight_line_transects.nc')
    m.prep_interp_to_raw_obs()
    print (m.save_append)

    for ind in range(0,100):

        m.ind = ind
        print ('ind: ', ind)

        # interpolate to uniform grid
        m.interp_to_raw_obs_path(random_offset=True, load_offset=True)
        m.interp_raw_obs_path_to_uniform_grid()

    # parallel save name
    m.save_append = m.save_append + '_shift_' + str(int(1/lon_shift))
    print (m.save_append)
    for ind in range(0,100):

        m.ind = ind
        print ('ind: ', ind)

        # interpolate to uniform grid
        m.interp_to_raw_obs_path(random_offset=True, load_offset=True,
                                 parallel_offset=lon_shift)
        m.interp_raw_obs_path_to_uniform_grid()

def calculate_buoyancy_gradients_across_straight_line_paths(fn, dist):

    # set paths
    prep = 'GliderRandomSampling/'
    path = config.data_path() + case + '/' + prep 

    # get path 1
    kwargs = dict(clobber=True,mode='a')
    g1 = xr.open_dataset(path + fn + '.nc', backend_kwargs=kwargs)

    # get path 2
    g2 = xr.open_dataset(path + fn + '_shift_' + dist + '.nc')

    # constants
    g = 9.81
    rho_0 = 1027 

    # distance between tracks
    lon0 = g1.isel(distance=100, ctd_depth=0, sample=0).lon
    lon1 = g2.isel(distance=100, ctd_depth=0, sample=0).lon
    lat0 = g1.isel(distance=100, ctd_depth=0, sample=0).lat
    lat1 = g2.isel(distance=100, ctd_depth=0, sample=0).lat


    # get haversine distance between tracks
    # haversine should perhaps be outside of the model_object
    m = model_object.model('EXP10')
    dx = m.haversine(lon0, lat0, lon1, lat1)
   
    # buoyancy gradient
    b_1 = g * (1 - g1.rho / rho_0)
    b_2 = g * (1 - g2.rho / rho_0)
    dbdtransect = np.abs((b_1 - b_2) / dx)

    # restrict to ml
    g1['b_x_ct_' + dist + '_ml'] = dbdtransect.where(g1.deptht < g1.mld)
 
    # save
    g1.to_netcdf(path + fn + '.nc')

def save_interpolated_transects_to_one_file(case, fn, n=100, rotation=None,
                                            add_transects=False):
    '''
    this needs adjusting
    currently has a conditional statement for get_transects that is
    not used
    method relies on orignal data already containing transects

    Transects required for spectra and geom. These plotting scripts have
    in-built routines for adding transects. Better to add this here?

    Combine is required for bootstrapping. Need to check if the calcs
    include the mesoscale transect.
    '''

    # set paths
    prep = 'GliderRandomSampling/' + fn
    data_path = config.data_path() + case + '/' + prep 


    if rotation:
        rotation_label = 'rotate_' + str(rotation) + '_' 
        rotation_rad = np.radians(rotation)
    else:
        rotation_label = ''
        rotation_rad = rotation # None type 

    sample_list = [data_path + '_' + rotation_label +
                   str(i).zfill(2) + '.nc' for i in range(n)]

    sample_set = []
    for i in range(n):
        print ('sample: ', i)
        sample = xr.open_dataset(sample_list[i],
                                 decode_times=False)
        sample['lon_offset'] = sample.attrs['lon_offset']
        sample['lat_offset'] = sample.attrs['lat_offset']
        sample = sample.set_coords(['lon_offset','lat_offset',
                                    'time_counter'])
        sample = sample.assign_coords({'sample': i + 1})

        if add_transects:
        # this removes n-s transect!
        # hack because transect doesn't currently take 2d-ds (1d-da only)
            b_x_ml_transect = get_transects(
                               sample.b_x_ml.isel(ctd_depth=10),
                               offset=True, rotation=rotation_rad,
                               method='find e-w')
            sample = sample.assign_coords(
              {'transect': b_x_ml_transect.transect.reset_coords(drop=True),
               'vertex'  : b_x_ml_transect.vertex.reset_coords(drop=True)})

        sample_set.append(sample.expand_dims('sample'))
    samples=xr.concat(sample_set, dim='sample')
    samples.to_netcdf(data_path + rotation_label.rstrip('_') + '.nc')

def restrict_bg_norm_to_mld(remove=False, append='', interp_dist=1000,
                            transects=False):
    ''' 
    Fix mistake made when adding bg_norm to glider samples.
    Variable was taken over full depth rather than being restricted
    to mld.
    '''        

    m = model_object.model('EXP10')

    m.save_append = 'interp_' + str(interp_dist) + append
    if remove:
        m.save_append = m.save_append + '_' + remove
    if transects:
        m.save_append = m.save_append + '_pre_transect'

    m.restrict_bg_norm_to_mld()


if __name__ == '__main__':

  
    dask.config.set(scheduler='single-threaded')

    #glider_sample_parallel_straight_line_paths()

    case='EXP10'
    fn = 'glider_uniform_interp_1000_parallel_transects'
    #save_interpolated_transects_to_one_file(case, fn)
    calculate_buoyancy_gradients_across_straight_line_paths(fn, '12')


    ######glider_sampling('EXP10', interp_dist=1000, transects=True)
    ######glider_sampling('EXP10', remove='every_2',
    ######                interp_dist=1000, transects=True)
    ######glider_sampling('EXP10', remove='every_4',
    ######                interp_dist=1000, transects=True)
    #glider_sampling('EXP10', remove='every_2_and_dive',
    #                interp_dist=1000, transects=True)
    #glider_sampling('EXP10', remove='every_2_and_climb',
    #                interp_dist=1000, transects=True)
    #glider_sampling('EXP10', remove='every_4_and_dive',
    #                interp_dist=1000, transects=True)

    # done 27th Oct
#    glider_sampling('EXP10', remove='every_3_and_climb',
#                    interp_dist=1000, transects=False)
    # do last 15
    #glider_sampling('EXP10', remove='every_2_and_climb',
    #                interp_dist=1000, transects=False)

    # not done
    #glider_sampling('EXP10', remove='every_4_and_climb',
    #                interp_dist=1000, transects=False)
    #glider_sampling('EXP10', remove='every_8_and_climb',
    #                interp_dist=1000, transects=False)

    #glider_sampling('EXP10', remove='every_3',
    #                interp_dist=1000, transects=False)
#    glider_sampling('EXP10', remove=False, append='interp_2000', 
#                    interp_dist=2000, transects=False, rotate=False)
    ###
    #north_limit=-59.9858036
    ###
