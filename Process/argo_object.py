import xarray as xr
import config
import numpy as np
import matplotlib.pyplot as plt


class argo(object):
    ''' get argo object and process '''
 
    def __init__(self, climatology):
        self.data_path = '/storage/silver/SO-CHIC/Ryan/Argo/'
        if climatology:
            self.clim = True
            self.ds = xr.open_dataset(self.data_path +
                                  'Argo_mixedlayers_monthlyclim_03172021.nc')
            # alias lat lon labels
            self.ds = self.ds.rename(lon='profilelon')
            self.ds = self.ds.rename(lat='profilelat')
        else:
            self.clim = False
            self.ds = xr.open_dataset(self.data_path +
                                  'Argo_mixedlayers_all_03172021.nc')
      
            # format argo time
            time_to_1970 = np.datetime64('1970-01-01') -\
                           np.datetime64('0000-01-01')
            days_to_1970 = time_to_1970.astype('float64')
            self.ds['profiledate'] = self.ds.profiledate - days_to_1970
            self.ds.profiledate.attrs['units'] = 'days since 1970-01-01'
            self.ds.profiledate.attrs['calendar'] = 'gregorian'
            self.ds = xr.decode_cf(self.ds)

    def cut_to_glider_dates(self):
        ''' cut dates to glider period '''
        
        glider = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
        date0 = glider.ctd_time.min()
        date1 = glider.ctd_time.max()

        self.ds = self.ds.where((self.ds.profiledate > date0) &
                                (self.ds.profiledate < date1), drop=True)

    def cut_to_sochic_patch(self):
        ''' cut area to match sochic model region '''

        grid = xr.open_dataset(config.data_path() +
                              'ORCA/coordinates_subset.nc', decode_times=False)
        lon0 = grid.nav_lon.min()
        lon1 = grid.nav_lon.max()
        lat0 = grid.nav_lat.min()
        lat1 = grid.nav_lat.max()

        self.ds = self.ds.where((self.ds.profilelon > lon0) &
                                (self.ds.profilelon < lon1) &
                                (self.ds.profilelat > lat0) &
                                (self.ds.profilelat < lat1), drop=True)

    def mean_by_day(self):
        ''' collect each day by averaging by day '''

        self.ds = self.ds.set_coords('profiledate')
        self.ds = self.ds.swap_dims({'iNPROF':'profiledate'})
        self.ds = self.ds.sortby('profiledate')
        self.ds = self.ds.resample(profiledate='1D').mean()
        #grouped = self.ds.groupby('profiledate.dayofyear')
        #self.ds = grouped.mean()
        print (self.ds)

    def mean_by_month(self):
        self.ds = self.ds.mean(['iLAT','iLON'], skipna=True)

    def save_processed_mld(self):
        ''' save lateral mean of all data '''

        if self.clim:
            self.ds.to_netcdf(
                      '/storage/silver/SO-CHIC/Ryan/Argo/argo_giddy_clim.nc')
        else:
            self.ds.to_netcdf('/storage/silver/SO-CHIC/Ryan/Argo/argo_giddy.nc')

def cut_argo_time_specific():
    m = argo()
    m.cut_to_sochic_patch()
    m.cut_to_glider_dates()
    m.mean_by_day()
    m.save_processed_mld()

def cut_argo_time_climatology():
    m = argo(climatology=True)
    m.cut_to_sochic_patch()
    m.mean_by_month()
    m.save_processed_mld()
cut_argo_time_climatology()
