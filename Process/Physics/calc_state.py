import model_object
import xarray as xr
import config
import gsw
import dask

class state(model_object.model):
    """
    process state variables
    """

    def __init__(self, case="EXP10"):

        super().__init__(case)

    def load_gsw(self):
        ''' load absolute salinity and conservative temperature '''

        #self.ds = {}
        #self.file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
        path = self.data_path + self.file_id + 'gsw.nc'
        self.ds['gsw'] = xr.open_dataset(path, chunks={'time_counter':10})

    def merge_state(self):
        ''' merge all state variables into one file '''

        grid_T = xr.open_dataset(self.data_path + 
                                 'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc',
                                    chunks={'time_counter':1})
        alpha = xr.open_dataarray(self.data_path + 'alpha.nc',
                                    chunks={'time_counter':1})
        beta = xr.open_dataarray(self.data_path + 'beta.nc',
                                    chunks={'time_counter':1})
        absolute_salinity = xr.open_dataarray(self.data_path + 
                                    'absolute_salinity.nc',
                                    chunks={'time_counter':1})
        conservative_temperature = xr.open_dataarray(self.data_path + 
                                        'conservative_temperature.nc',
                                        chunks={'time_counter':1})
        self.ds = xr.merge([alpha, beta, absolute_salinity,
                           conservative_temperature,
                           grid_T.mldr10_3])

        # make grid regular
        self.x_y_to_lat_lon()
 
        self.ds.to_netcdf('state.nc')

    def get_pressure(self, save=False):
        ''' calculate pressure from depth '''
        if self.loaded_p:
            print ('p already loaded') 
        else:
            self.loaded_p = True
            data = self.ds['grid_T']
            self.p = gsw.p_from_z(-data.deptht, data.nav_lat)
            self.p.name = 'p'
            if save:
                self.p.to_netcdf(self.data_path + self.file_id + 'p.nc')

    def get_conservative_temperature(self, save=False):
        ''' calulate conservative temperature '''
        data = self.ds['grid_T'].chunk({'time_counter':1})
        #self.cons_temp = gsw.conversions.CT_from_pt(data.vosaline,
        #                                            data.votemper)
        self.cons_temp = xr.apply_ufunc(gsw.conversions.CT_from_pt,
                                        data.vosaline, data.votemper,
                                        dask='parallelized',
                                        output_dtypes=[data.vosaline.dtype])
        #self.cons_temp.compute()
        self.cons_temp.name = 'cons_temp'
        if save:
            self.cons_temp.to_netcdf(self.data_path + self.file_id
                                   + 'conservative_temperature.nc',)

    def get_absolute_salinity(self, save=False):
        ''' calulate absolute_salinity '''
        self.get_pressure()
        data = self.ds['grid_T'].chunk({'time_counter':1})
        #self.abs_sal = gsw.conversions.SA_from_SP(data.vosaline, 
        #                                          self.p,
        #                                          data.nav_lon,
        #                                          data.nav_lat)
        self.abs_sal = xr.apply_ufunc(gsw.conversions.SA_from_SP,data.vosaline, 
                                      self.p, data.nav_lon, data.nav_lat,
                                      dask='parallelized', 
                                      output_dtypes=[data.vosaline.dtype])
        #self.abs_sal.compute()
        self.abs_sal.name = 'abs_sal'
        if save:
            self.abs_sal.to_netcdf(self.data_path + self.file_id 
                                + 'absolute_salinity.nc')


    def get_alpha_and_beta(self, save=False):
        ''' calculate the themo-haline contaction coefficients '''
        #self.open_ct_as_p()
        alpha = xr.apply_ufunc(gsw.density.alpha,
                                          self.ds['gsw'].abs_sal,
                                          self.ds['gsw'].cons_temp,
                                          self.ds['gsw'].p,
                                          dask='parallelized',
                                   output_dtypes=[self.ds['gsw'].abs_sal.dtype])
        beta = xr.apply_ufunc(gsw.density.beta,
                                         self.ds['gsw'].abs_sal,
                                         self.ds['gsw'].cons_temp,
                                         self.ds['gsw'].p,
                                          dask='parallelized',
                                  output_dtypes=[self.ds['gsw'].abs_sal.dtype])

        if save:
            alpha.to_netcdf(config.data_path() + self.file_id + 'alpha.nc')
            beta.to_netcdf(config.data_path() + self.file_id + 'beta.nc')

    def get_rho_theta(self):
        '''
        calculate potential density from conservative temperature and
        absolute salinity    
        '''
        
        # load temp, sal, alpha, beta
        gsw_file = xr.open_dataset(self.data_path + self.file_id +  'gsw.nc',
                              chunks={'time_counter':1})
        ct = gsw_file.cons_temp
        a_sal = gsw_file.abs_sal

        rho = xr.apply_ufunc(gsw.density.sigma0, a_sal, ct,
                             dask='parallelized', output_dtypes=[a_sal.dtype]
                             ) + 1000

        # save
        rho.name = 'rho_theta'
        rho.to_netcdf(self.data_path + self.file_id + 'rho.nc')

    def get_rho(self):
        '''
        calculate in-situ density from conservative temperature and
        absolute salinity    
        '''
        
        # load temp, sal, alpha, beta
        gsw_file = xr.open_dataset(self.data_path + self.file_id +  'gsw.nc',
                              chunks={'time_counter':1})
        ct = gsw_file.cons_temp
        a_sal = gsw_file.abs_sal

        rho = xr.apply_ufunc(gsw.density.rho, a_sal, ct,
                             dask='parallelized', output_dtypes=[a_sal.dtype]
                             )

        # save
        rho.name = 'rho_in_situ'
        rho.to_netcdf(self.data_path + self.file_id + 'rho_in_situ.nc')

    def save_all_gsw(self):
        ''' save p, conservative temperature and absolute salinity to netcdf '''

        self.get_pressure()
        self.get_conservative_temperature()
        self.get_absolute_salinity()
        gsw = xr.merge([self.p, self.cons_temp, self.abs_sal])
        gsw.to_netcdf(self.data_path + self.file_id + 'gsw.nc')

if __name__ == '__main__':

    def get_rho(case):
        m = model(case)
        m.get_rho()

    def save_alpha_and_beta(case):
        m = model(case)
        m.load_gridT_and_giddy()
        m.load_gsw()
        print (m.ds)
        m.get_alpha_and_beta(save=True)

    get_rho('EXP10')
