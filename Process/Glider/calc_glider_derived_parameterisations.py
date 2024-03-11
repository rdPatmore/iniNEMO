import xarray as xr
import config

class glider_derived_parameterisations(object):
    """
    calculate parameterisations using glider sampled model data
    """

    def __init__(self, case):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + "/"

        self.file_id = "/SOCHIC_PATCH_3h_20121209_20130331_"

    def get_glider_samples(self):
        """ load processed glider samples """

        # files definitions
        prep = "GliderRandomSampling/glider_uniform_interp_1000.nc"

        # get samples
        self.samples = xr.open_dataset(self.data_path + prep,
                                       decode_times=False, chunks=-1)

    def qmel(self):
        """
        Calculate the heat flux assocated with baroclinic instabilites
        """
 
        print (self.samples)
        bx = self.samples.b_x_ml
        H = self.samples.mld
        Cp = 3998
        g=9.81
        
        #0.006 * self.samples.b_x_ml**2 *  


gdp = glider_derived_parameterisations("EXP10")
gdp.get_glider_samples()
gdp.qmel()
