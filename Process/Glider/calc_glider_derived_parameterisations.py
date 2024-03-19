import xarray as xr
import config
import matplotlib.pyplot as plt

class glider_derived_parameterisations(object):
    """
    calculate parameterisations using glider sampled model data
    """

    def __init__(self, case):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + "/"

        self.file_id = "/SOCHIC_PATCH_3h_20121209_20130331_"
        self.prep = "GliderRandomSampling/"

    def get_glider_samples(self):
        """ load processed glider samples """

        # files definitions

        # get samples
        kwargs = dict(clobber=True, mode="a")
        self.samples = xr.open_dataset(self.data_path + self.prep +
                                      "glider_uniform_interp_1000.nc",
                                       decode_times=False, chunks=-1,
                                       backend_kwargs=kwargs)

    def save_qmel(self):
        """
        Calculate the heat flux assocated with baroclinic instabilites
        """
 
        # define parameters
        bx = self.samples.b_x_ml.mean("ctd_depth", skipna=True)
        H = self.samples.mld
        Cp = 3998
        g = 9.81
        f = -1.2638e-4      # mean of model
        alpha = 8.789123e-5 # mean of model
        rho0 = 1026
        
        # calculate mle
        self.samples["qmle"] = (0.006 * bx**2 * H**2 * Cp * rho0 /  
                               (f * alpha * g)).load()

        # save
        self.samples.to_netcdf(self.data_path + self.prep +
                               "glider_uniform_interp_1000_with_qmle.nc")

gdp = glider_derived_parameterisations("EXP10")
gdp.get_glider_samples()
gdp.save_qmel()
