import model_object
import numpy as np

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
    m.load_gridT_and_giddy(bg=True)

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

    # initialise object
    m = model_object.model('EXP10')

    # get data and set file naming
    m.load_gridT_and_giddy(bg=True)
    m.save_append = 'interp_' + str(interp_dist) + '_parallel_transects'

    for ind in range(0,100):

        m.ind = ind
        print ('ind: ', ind)

        # interpolate to uniform grid
        m.interp_to_raw_obs_path(random_offset=True, load_offset=True)
        m.interp_raw_obs_path_to_uniform_grid(ind=ind)

def combine_glider_samples(case, remove=False, append='', interp_dist=1000,
                    transects=False, south_limit=None, north_limit=None,
                    rotate=False, rotation=np.pi/2):
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

    m = model(case)
    m.interp_dist=interp_dist
    m.transects=transects
    #m.load_gridT_and_giddy()
    m.append = append

    # reductions of nemo domain
    m.south_limit = south_limit
    m.north_limit = north_limit

    m.save_interpolated_transects_to_one_file(n=100, rotation=None)

#    combine_glider_samples('EXP10',
#                           append='interp_1000', 
#                           interp_dist=1000, transects=False)
    #combine_glider_samples('EXP10', remove=False,
    #                       append='interp_1000_north_patch', 
    #                       interp_dist=1000, transects=False, rotate=False)
glider_sampling('EXP10', interp_dist=1000, transects=False)
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
