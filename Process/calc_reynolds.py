import xarray as xr
import config
import matplotlib.pyplot as plt
import dask

class reynolds(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_1h_20121209_20121211_'
        self.nc_preamble = config.data_path() + case +  self.file_id

    def get_time_mean_vels(self):
        #kwargs = {'chunks':{'depthu':1} ,'decode_cf':False} 
        kwargs = {'decode_cf':False} 
        #with xr.open_dataset(self.nc_preamble + 'grid_U.nc', **kwargs).uo as u:
        with xr.open_dataset(self.nc_preamble + 'uvel_mld.nc', **kwargs).uo as u:
            u_mean = u.mean(['time_counter']).load()
            u_mean.to_netcdf(self.nc_preamble + 'uvel_mld_mean.nc')

        #kwargs = {'chunks':{'depthv':-1} ,'decode_cf':False} 
        kwargs = {'decode_cf':False} 
        with xr.open_dataset(self.nc_preamble + 'vvel_mld.nc', **kwargs).vo as v:
            v_mean = v.mean(['time_counter']).load()
            v_mean.to_netcdf(self.nc_preamble + 'vvel_mld_mean.nc')

    def get_u_prime(self):
        # load
        #kwargs = {'chunks':{'depthu':-1} ,'decode_cf':False} 
        kwargs = {'decode_cf':False} 
        u = xr.open_dataset(self.nc_preamble + 'uvel_mld.nc', **kwargs).uo
        u_mean = xr.load_dataarray(self.nc_preamble + 'uvel_mld_mean.nc')

        # reynolds
        u_prime = u - u_mean
        # save
        u_prime.to_netcdf(self.nc_preamble + 'uvel_mld_rey.nc')

    def get_v_prime(self):
        # load
        #kwargs = {'chunks':{'depthv':-1} ,'decode_cf':False} 
        kwargs = {'decode_cf':False} 
        v = xr.open_dataset(self.nc_preamble + 'vvel_mld.nc', **kwargs).vo
        v_mean = xr.load_dataarray(self.nc_preamble + 'vvel_mld_mean.nc')

        # reynolds
        v_prime = v - v_mean

        # save
        v_prime.to_netcdf(self.nc_preamble + 'vvel_mld_rey.nc')

    def get_lateral_vel_primes(self):
        # get time means
        self.get_time_mean_vels()

        # get primes
        self.get_u_prime()
        self.get_v_prime()

    def get_time_mean_mom(self, vec):
        #kwargs = {'chunks':{'depth' + vec:1} ,'decode_cf':False} 
        mn = 'mom' + vec
        kwargs = {'decode_cf':False} 
        with xr.open_dataset(self.nc_preamble + mn + '_mld.nc', **kwargs) as mom:
            mean = mom.mean(['time_counter']).load()
            mean.to_netcdf(self.nc_preamble + mn + '_mld_mean.nc')

    def get_momu_prime(self):
        # load
        kwargs = {'decode_cf':False} 
        u = xr.open_dataset(self.nc_preamble + 'momu_mld.nc', **kwargs)
        u_mean = xr.load_dataset(self.nc_preamble + 'momu_mld_mean.nc')

        # reynolds
        u_prime = u - u_mean
        # save
        u_prime.to_netcdf(self.nc_preamble + 'momu_mld_rey.nc')

    def get_momv_prime(self):
        # load
        kwargs = {'decode_cf':False} 
        v = xr.open_dataset(self.nc_preamble + 'momv_mld.nc', **kwargs)
        v_mean = xr.load_dataset(self.nc_preamble + 'momv_mld_mean.nc')

        # reynolds
        v_prime = v - v_mean

        # save
        v_prime.to_netcdf(self.nc_preamble + 'momv_mld_rey.nc')

    def get_lateral_mom_primes(self):
        # get time means
        #self.get_time_mean_mom('u')
        #self.get_time_mean_mom('v')

        # get primes
        #self.get_momu_prime()
        self.get_momv_prime()

if __name__ == '__main__':
    import time
    start = time.time()
    m = reynolds('EXP90')
    #m.get_lateral_vel_primes()
    m.get_lateral_mom_primes()
    end = time.time()
    print('time elapsed (minutes): ', (end - start)/60)
