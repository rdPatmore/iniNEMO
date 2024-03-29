import xarray as xr
import config
import numpy as np


class satellite(object):
    ''' get model object and process '''
 
    def __init__(self, year):
        self.year = year
        self.data_path = config.data_path() + 'SeaIce/'
        self.ds = xr.open_dataset(self.data_path +
                                  'seaice_conc_daily_sh_' + year + '_v04r00.nc')
        self.ds = self.ds.drop('projection')
                                 #decode_times=False, engine='h5netcdf')

    def cut_to_sochic_patch(self):
        ''' cut area to match sochic model region '''

        grid = xr.open_dataset(config.data_path() +
                              'ORCA/coordinates_subset.nc', decode_times=False)
        lon0 = grid.nav_lon.min()
        lon1 = grid.nav_lon.max()
        lat0 = grid.nav_lat.min()
        lat1 = grid.nav_lat.max()

        self.ds = self.ds.where((self.ds.longitude > lon0) &
                                (self.ds.longitude < lon1) &
                                (self.ds.latitude  > lat0) &
                                (self.ds.latitude  < lat1), drop=True)
        self.ds['icepres'] = xr.where(self.ds.cdr_seaice_conc > 0, 1, 0)
        self.ds.to_netcdf(self.data_path + 
                       'seaice_conc_daily_sh_' + self.year + '_sochic_patch.nc')

    def save_area_mean_all(self):
        ''' save lateral mean of all data '''

        # create weights
        weights = np.cos(np.deg2rad(self.ds.latitude))
        weights.name = 'weights'

        # apply weights
        weights = weights.fillna(0)

        for key in self.ds.keys():
            da = self.ds[key].weighted(weights)
            print (key)
            self.ds[key] = da.mean(['x','y'])
            ds = self.ds.rename({key: key + '_mean'})
        ds = ds.swap_dims({'tdim':'time'})
        ds.to_netcdf(self.data_path + 'seaice_conc_daily_sh_' 
                     + self.year + '_mean.nc')

    def save_area_std_all(self):
        ''' save lateral standard deviation of all data '''

        ds = self.ds.std(['x','y']).load()
        for key in ds.keys():
            ds = ds.rename({key: key + '_std'})
        ds = ds.swap_dims({'tdim':'time'})
        ds.to_netcdf(self.data_path + 'seaice_conc_daily_sh_' 
                     + self.year + '_std.nc')


m = satellite('2014')
m.cut_to_sochic_patch()
m.save_area_mean_all()
