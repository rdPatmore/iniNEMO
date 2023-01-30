import xarray as xr
import config
import matplotlib.pyplot as plt

class cori_err(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_1h_20121209_20121211_'
        self.preamble = config.data_path() + case +  self.file_id
        self.path = config.data_path() + case + '/'

    def im1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(x=-1, roll_coords=False)

    def ip1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(x=1, roll_coords=False)

    def jm1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(y=-1, roll_coords=False)

    def jp1(self, var):
        ''' rolling opperations: roll west '''

        return var.roll(y=1, roll_coords=False)

    def calc_cori_err_30(self):
        ''' calculate the gridding error arrising due to cori gridding'''

        uvel = xr.open_dataset(self.preamble + 'uvel_30.nc').uo
        vvel = xr.open_dataset(self.preamble + 'vvel_30.nc').vo
        #e3u = xr.open_dataset(self.preamble + 'uvel_30.nc').e3u
        #e3v = xr.open_dataset(self.preamble + 'vvel_30.nc').e3v
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc').isel(z=13)
        ff_f = xr.open_dataset(self.path + 'domain_cfg.nc').ff_f

        uflux = uvel * cfg.e2u * cfg.e3u_0
        vflux = vvel * cfg.e1v * cfg.e3v_0

        f3_ne = (         ff_f  + self.im1(ff_f) + self.jm1(ff_f))
        f3_nw = (         ff_f  + self.im1(ff_f) + self.im1(self.jm1(ff_f))) 
        f3_se = (         ff_f  + self.jm1(ff_f) + self.im1(self.jm1(ff_f))) 
        f3_sw = (self.im1(ff_f) + self.jm1(ff_f) + self.im1(self.jm1(ff_f))) 
        
        uPVO = (1/12.0)*(1/cfg.e1u)*(1/cfg.e3u_0)*( 
                                             f3_ne      * vflux 
                                  + self.ip1(f3_nw) * self.ip1(vflux)
                                  +          f3_se  * self.jm1(vflux)
                                  + self.ip1(f3_sw) * self.ip1(self.jm1(vflux)) )
        
        vPVO = -(1/12.0)*(1/cfg.e2v)*(1/cfg.e3v_0)*(
                                     self.jp1(f3_sw) * self.im1(self.jp1(uflux))
                                   + self.jp1(f3_se) * self.jp1(uflux)
                                   +          f3_nw  * self.im1(uflux)
                                   +          f3_ne  * uflux )

        uPVO.to_netcdf(self.preamble + 'utrd_pvo_bta_30.nc')
        vPVO.to_netcdf(self.preamble + 'vtrd_pvo_bta_30.nc')

if __name__ == '__main__':
    c = cori_err('TRD00')
    c.calc_cori_err_30()
