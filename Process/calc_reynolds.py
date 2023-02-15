import xarray as xr
import config
import matplotlib.pyplot as plt
import dask

class reynolds(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_1h_20121209_20121211_'
        self.nc_preamble = config.data_path() + case +  self.file_id

    def get_time_mean_lat_vels(self):
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

    def get_time_mean_w_vels(self):
        kwargs = {'decode_cf':False} 
        with xr.open_dataset(self.nc_preamble + 'wvel_mld.nc', **kwargs).wo as w:
            w_mean = w.mean(['time_counter']).load()
            w_mean.to_netcdf(self.nc_preamble + 'wvel_mld_mean.nc')

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

    def get_w_prime(self):
        # load
        kwargs = {'decode_cf':False} 
        w = xr.open_dataset(self.nc_preamble + 'wvel_mld.nc', **kwargs).wo
        w_mean = xr.load_dataarray(self.nc_preamble + 'wvel_mld_mean.nc')

        # reynolds
        w_prime = w - w_mean

        # save
        w_prime.to_netcdf(self.nc_preamble + 'wvel_mld_rey.nc')

    def get_lateral_vel_primes(self):
        # get time means
        self.get_time_mean_lat_vels()

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
        self.get_time_mean_mom('u')
        self.get_time_mean_mom('v')

        # get primes
        self.get_momu_prime()
        self.get_momv_prime()


    def get_rho_prime(self):
        # get time mean rho
        kwargs = {'decode_cf':False} 
        with xr.open_dataset(self.nc_preamble + 'rho_mld.nc', **kwargs) as rho:
            mean = rho.mean(['time_counter']).load()
            mean.to_netcdf(self.nc_preamble + 'rho_mld_mean.nc')

        # load
        kwargs = {'decode_cf':False} 
        rho = xr.open_dataset(self.nc_preamble + 'rho_mld.nc', **kwargs)
        rho_mean = xr.load_dataset(self.nc_preamble + 'rho_mld_mean.nc')

        # reynolds
        rho_prime = rho - rho_mean

        # save
        rho_prime.to_netcdf(self.nc_preamble + 'rho_mld_rey.nc')

    def reduce_time_resolution(self):
        ''' reduce time interval of data '''

            if interval == '1h':
                split = [0]
                f = f.isel(time_counter=ds.time_counter.dt.minute.isin(split))
            elif interval == '30mi':
                split = [0,30]
                f = f.isel(time_counter=ds.time_counter.dt.minute.isin(split))

    def get_time_mean(self, fname, fname_out, var=None):
        ''' general routine for finding means '''

        kwargs = {'decode_cf':False} 
        with xr.open_dataset(self.nc_preamble + fname + '.nc', **kwargs) as f:
            if var:
               f = f[var]
            if fname in ['grid_T_dep_30','grid_T']:
               f = f.set_coords('time_instant')

            # mean
            f_mean = f.mean('time_counter').load()
            
            # save
            f_mean.to_netcdf(self.nc_preamble + fname_out + '_mean.nc')

    def get_primes(self, fname, fname_out, var=None):
        ''' general routine for finding primes '''

        # load
        kwargs = {'decode_cf':False} 
        f = xr.open_dataset(self.nc_preamble + fname + '.nc')
        f_mean = xr.load_dataset(self.nc_preamble + fname + '_mean.nc')
        if fname in ['grid_T_30','grid_T']:
           f = f.set_coords('time_instant')

        # reynolds
        f_prime = f - f_mean

        # save
        f_prime.to_netcdf(self.nc_preamble + fname_out + '_rey.nc')

    def get_means_all(self):

        # means
        self.get_time_mean('uvel_dep_30', 'uvel_dep_30')
        self.get_time_mean('vvel_dep_30', 'vvel_dep_30')
        self.get_time_mean('momu_dep_30', 'momu_dep_30')
        self.get_time_mean('momv_dep_30', 'momv_dep_30')
        self.get_time_mean('grid_T_dep_30' , 'grid_T_dep_30')

        # for buoyancy flux
        self.get_time_mean('grid_T' , 'grid_T')
        self.get_time_mean('grid_W' , 'grid_W')
        self.get_time_mean('rhoW' , 'rhoW')

    def get_primes_all(self):

        # means
        self.get_primes('uvel_dep_30', 'uvel_dep_30')
        self.get_primes('momu_dep_30', 'momu_dep_30')
        self.get_primes('momv_dep_30', 'momv_dep_30')
        self.get_primes('grid_T_dep_30' , 'grid_T_dep_30')

        # for buoyancy flux
        self.get_primes('grid_T' , 'grid_T')
        self.get_primes('grid_W' , 'grid_W')
        self.get_primes('rhoW' , 'rhoW')

if __name__ == '__main__':
    import time
    start = time.time()
    m = reynolds('TRD00')
    #m.get_time_mean('uvel_30', 'uvel_30')
    #m.get_primes('uvel_30', 'uvel_30')
    #m.get_time_mean('grid_T_30', 'grid_T_30')
    m.get_primes('grid_T_30', 'grid_T_30')
    #m.get_primes_all()

    #m.get_lateral_vel_primes()
    #m.get_lateral_mom_primes()
    #m.get_time_mean_w_vels()
    #m.get_w_prime()
    #m.get_rho_prime()
    end = time.time()
    print('time elapsed (minutes): ', (end - start)/60)
