import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np

class surface_forcing(object):

    def __init__(self, case):
        self.case = case
        self.save_path = config.data_path() + self.case + '/'

        self.ds = xr.open_mfdataset(self.save_path + 'DFS*.nc',
                                    chunks={'time':1})
    def calc_mod_tauw(self):
        ''' calculate the wind speed '''
       
        ws = (self.ds.u10 ** 2 + self.ds.v10 ** 2) ** 0.5
        ws.name = 'mod_tauw'
        #ws = xr.merge([ws, self.ds.nav_lat, self.ds.nav_lon])
        ws.to_netcdf(self.save_path + 'mod_tauw.nc')

    def calc_wind_speed_means(self):
        ''' 
              calculate the mean wind speed for 
              entire, nothern and southern part of domain
        '''

        tau = xr.open_dataset(self.save_path + 'mod_tauw.nc',
                                chunks={'time':1})
       
        # get north and south
        mid_lat = self.ds.nav_lat.quantile(0.5, ['X','Y'])
        #taun = tau.where(self.ds.nav_lat>=mid_lat,drop=True).drop_vars('quantile')
        #taus = tau.where(self.ds.nav_lat<mid_lat,drop=True).drop_vars('quantile')
        taun = tau.where(self.ds.nav_lat>=mid_lat,drop=True).to_dataarray()
        taus = tau.where(self.ds.nav_lat<mid_lat,drop=True).to_dataarray()
        print (taun)

        # means
        taun_mean = taun.mean(['X','Y'])
        taus_mean = taus.mean(['X','Y'])
        tau_mean  = tau.mean(['X','Y'])

        # standard deviations
        taun_std = taun.std(['Y','Y'])
        taus_std = taus.std(['Y','Y'])
        tau_std  = tau.std(['Y','Y'])

        # rename
        print (taun_mean)
        taun_mean.name = 'taun_mean'
        taus_mean.name = 'taus_mean'
        tau_mean.name  = 'tau_mean'
        taun_std.name  = 'taun_std'
        taus_std.name  = 'taus_std'
        tau_std.name   = 'tau_std'

        # save
        tau_stats = xr.merge([taun_mean, taus_mean, tau_mean,
                              taun_std,  taus_std,  tau_std])

        tau_stats.to_netcdf(self.save_path + 'tau_stats.nc')

sf = surface_forcing('EXP02')
#sf.calc_mod_tauw()
sf.calc_wind_speed_means()
