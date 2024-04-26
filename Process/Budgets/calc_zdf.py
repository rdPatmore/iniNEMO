import xarray as xr
import config
import dask
from dask.diagnostics import ProgressBar

class zdf(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_15mi_20121209_20121211_'
        self.preamble = config.data_path() + case +  self.file_id
        self.path = config.data_path() + case + '/'

    def get_zdf(self, vec='u'):
        '''
        Get zdf with wind stress, bottom friction and top friction removed.
        '''

        chunks = dict(time_counter=1)
        mom = xr.open_dataset(self.preamble + 'mom{}.nc'.format(vec),
                              chunks=chunks)


        mom = mom.drop(['time_instant', 'time_instant_bounds'])
        mom = mom.set_coords(['area',
                                'bounds_nav_lat',
                                'bounds_nav_lon',
                                'h{}'.format(vec),
                                'time_counter_bounds',
                                'depth{}_bounds'.format(vec)])

        drag_terms = mom[vec + 'trd_tau2d'] \
                   + mom[vec + 'trd_bfr2d'] \
                   + mom[vec + 'trd_tfr2d']
        mom[vec + 'trd_zdf'] = mom[vec + 'trd_zdf'] - drag_terms

        
        for var in list(mom.keys()):
            chunks = (tuple(max(c) for c in mom[var].chunks))
            mom[var].encoding['chunksizes'] = chunks 


        with ProgressBar():
            mom.to_netcdf(self.preamble + 'mom{}_zdf.nc'.format(vec))


if __name__ == '__main__':
    zdf = zdf('TRD00')
    zdf.get_zdf(vec='v')
