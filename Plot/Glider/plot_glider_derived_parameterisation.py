import xarray as xr
import config
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np

class parameterisation(object):

    def __init__(self, case):

        self.data_path = config.data_path() + case
        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'

    def get_qmle(self):
        """
        Get parameterisation of the heat flux associated with baroclinic
        instabilities.
        """

        fn = "glider_uniform_interp_1000_with_qmle.nc"
        self.ds = xr.open_dataset(self.data_path +
                                    "/GliderRandomSampling/" + fn,
                                    chunks=-1)

        self.ds["time_counter"] = self.ds.time_counter.mean(
                                                         ["sample","ctd_depth"])
        self.ds = self.ds.swap_dims({"distance":"time_counter"})

        self.qmle = self.ds.qmle#.rolling(time_counter=168).mean()


    def get_N2_median(self, depth=None):
        """
        Get stratification for each patch sampled by a glider
        and find the median
        """

        fn = self.file_id + "N2_100m_patch_set_mean.nc"
        if depth=="mld":
            print ("yesssss")
            fn = self.file_id + "N2_patch_set_mean.nc"
        #fn = self.file_id + "N2_patch_set_time_mean_space_quantile.nc"
        return xr.open_dataset(self.data_path +
                                    "/PatchSets/" + fn,
                                     chunks=-1).bn2
                                     #chunks=-1).bn2_rolling_mean

        #self.N2_median = self.N2_median.mean(["xi","yi"])

    def plot_qmle_N2_time_series(self):
        """
        Plot a time series of glider parameterised heat flux and stratification
        on separate panels
        """

        self.get_qmle()
        mld_N2 = self.get_N2_median(depth="mld")
        self.N2_median = self.get_N2_median()

        self.qmle = self.qmle.dropna("time_counter")

        # interpolate to model time
        self.qmle = self.qmle.interp(time_counter=self.N2_median.time_counter)

        fig, axs = plt.subplots(2, figsize=(7.2,4))
        plt.subplots_adjust()

        #self.qmle = self.qmle.isel(sample=1)
        #self.N2_median = self.N2_median.isel(sample=1)
        #mld_N2 = mld_N2.isel(sample=1)

        axs[0].plot(self.qmle.time_counter, self.qmle.T, c='blue', alpha=0.2)
        #axs[1].plot(self.N2_median.time_counter, self.N2_median.T,
        #                c='blue', label="100 m", alpha=0.2)
        axs[1].plot(mld_N2.time_counter, mld_N2.T,
                        c='red', label="mld", alpha=0.2)
        #axs[1].legend()
        axs[0].set_ylabel(r"$Q_{mle}$")
        axs[1].set_ylabel(r"$N^2$")
        axs[0].set_xlabel("time")
        axs[1].set_xlabel("time")

        plt.savefig("qmle_N2_time_series_all_sample.png", dpi=900)

    def plot_qmle_N2_cross_correlation(self):
        """
        Plot the cross correlation between the glider parameterised fluxes
        and the modelled mixed layer stratification
        """

        self.get_qmle()
        self.get_N2_median()

        self.qmle = self.qmle.dropna("time_counter")
        self.get_N2_median = self.N2_median.dropna("time_counter")

        # interpolate to model time
        self.qmle = self.qmle.interp(time_counter=self.N2_median.time_counter)

        fig, axs = plt.subplots(figsize=(3.2,4))
        plt.subplots_adjust()

        # calculate cross correlation
        cross_corr = []
        for sample in range(self.qmle.sample.size):
            print (sample)
            x = self.N2_median.isel(sample=sample)
            y = self.qmle.isel(sample=sample)
        
            mean=x.mean()
            var=np.var(x)
            xp=x-mean/var/len(x)
            mean=y.mean()
            var=np.var(y)
            yp=y-mean/var/len(y)
            corr=np.correlate(xp,yp,'full')[len(x) -1:]

            cross_corr = corr
            #cross_corr = np.ma.corrcoef(self.N2_sample, self.qmle_sample)
            #lags = sig.correlation_lags(len(self.qmle_sample),
            #                            len(self.N2_sample))
            #cross_corr /= np.nanmin(cross_corr) 
            print (cross_corr)
            plt.plot(cross_corr)
        plt.show()

        #cross_corr = np.array(cross_corr)

if __name__ == "__main__":
    par = parameterisation("EXP10")
    #par.plot_qmle_N2_cross_correlation()
    par.plot_qmle_N2_time_series()
