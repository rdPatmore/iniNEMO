import os, sys, glob
import numpy as np
import datetime
import xarray as xr

__author__      = "Nicolas Bruneau and Ryan Patmore"
__copyright__   = "Copyright 2018, NOC"
__email__       = "ryapat@noc.ac.uk"
__date__        = "2023-05"

class era5(object):
    """
    Generate ERA5 atmospheric forcing for NEMO
    So far to prduce a year by a year - need to be automated
    Loosly based on code by Nico
    """

    def __init__(self, pythonic=False):
        self.year_init = 1979                     ## First year to process
        self.year_end  = 2021                     ## Last one [included]
        self.east      =   19                     ## East Border
        self.west      =  -28                     ## West Border
        self.north     =   68                     ## North Border
        self.south     =   38                     ## South Border
        self.path_ERA5 = '/projectsa/NEMO/Forcing/ERA5/SURFACE_FORCING' 
        ## ROOT PATH OF ERA5 DATA
        ## WHERE TO EXTRACT YOUR REGION
        self.path_EXTRACT = '/projectsa/NEMO/ryapat/Extract' 
        ## NEMO FORCING
        self.path_FORCING = '/projectsa/NEMO/ryapat/Forcing'
        self.clean        = False               ## Clean extraction (longest bit)
        self.sph_ON       = False               ## Compute specific humidity or not
        # use python for interpolation in place of cdo - 4x slow but lower storage 
        self.pythonic = True  
        self.chunks={'time':50}

        self.var_path = { "10m_u_component_of_wind" : "u10"}
        #self.var_path = { "10m_u_component_of_wind" : "u10", \
        #             "10m_v_component_of_wind" : "v10", \
        #             "2m_temperature"          : "t2m", \
        #             "mean_sea_level_pressure" : "msl", \
        #             "mean_snowfall_rate"      : "msr" , \
        #         "mean_surface_downward_long_wave_radiation_flux"  : "msdwlwrf", \
        #         "mean_surface_downward_short_wave_radiation_flux" : "msdwswrf", \
        #             "mean_total_precipitation_rate" : "mtpr" }
        
        if self.sph_ON :
           self.var_path[ "surface_pressure"  ] = 'sp'
           self.var_path[ "2m_dewpoint_temperature" ] = 'd2m'

    #===================== INTERNAL FCTNS ========================
    
    def timeit(func):
        """ decorator for timing a function """ 

        def inner():
            t0 = datetime.datetime.now()
            func()
            t1 = datetime.datetime.now()
            print ('time elapsed = ', t1-t0)
            
        return inner
    
    def read_NetCDF(self, fname, KeyVar, chunks=None):
        """Read NetCDF file"""

        if "*" in fname: 
            lfiles = sorted( glob.glob( fname ) )
            ds = xr.open_mfdataset(lfiles, chunks=chunks, parallel=True,
                                   decode_times=False)
        else: 
            ds = xr.open_dataset(fname, chunks=chunks, decode_times=False)
    
        return ds[KeyVar]
    
    def set_attrs(self, ds):
        """ set global attributes for netcdf """
    
        fmt = "%Y-%m-%d %H:%M:%S"
        ds.attrs['Created'] = datetime.datetime.now().strftime(fmt)
        ds.attrs['Description'] = 'ERA5 Atmospheric conditions for AMM15 NEMO'
    
        return ds
    
    def cf_to_int_time(self, ds):
        ''' convert time units from cf decoded to int dtype '''
    
        ds['time'] = ds.time.astype(int) * 1e-9
        ds.time.attrs['units'] = 'seconds since 1970-01-01 00:00:00'
    
        return ds
    
    def extract(self, fin, fout, clean=True ) :
        if clean : os.system( "rm {0}".format( fout ) )
        if not os.path.exists( fout ) :
           command = "ncks -d latitude,{0},{1} -d longitude,{2},{3} {4} {5}".format(
                      np.float(South), np.float(North),
                      np.float(West), np.float(East), fin, fout )
           print (command)
           os.system( command )
    
    def interp_time(self, ds, fin, fout):
        """ 
        interpolate time to half timestep 
        cdo version of interpolation
        """
        if self.clean : os.system( "rm {0}".format( fout ) )
        if not os.path.exists( fout ) :
           fmt = "%Y-%m-%d"
           day0 = ds.time.dt.strftime(fmt)[0].values
           command = "cdo inttime,{0},{1},1hour {2} {3}".format(
                      day0, '00:30:00', fin, fout )
           print (command)
           os.system( command )
          
    def extract_cut_out(self, nameVar):
        for iY in range( self.year_init, self.year_end+1 ) :
            ## Files
            finput  = "{0}/{1}/{2}_{1}.nc".format(
              self.path_ERA5, self.dirVar, iY )
            foutput = "{2}/{0}_y{1}.nc".format(
              nameVar, iY, self.path_EXTRACT )
            ## Extract the subdomain
            self.extract(finput, foutput) 

    def interpolate_all(self, nameVar, foutInterp, pythonic=False):
        if not os.path.exists( foutInterp ) :
            if pythonic:
                ds = self.read_NetCDF(
                       "{1}/{0}_y*.nc".format(nameVar, self.path_EXTRACT), nameVar,
                          chunks=self.chunks)
    
                ## assume to be constant in time
                Time = ds.time.values
                dt  = (Time[1] - Time[0])#.astype('timedelta64[s]') 
                dt2 = dt / 2
                print ("dt", dt, dt2)
    
                ##---------- SOME PREPROCESSING -------------------------
                ## Add time step for last hour - copy the last input
                ## instantaneous field every hour. we center 
                ## it in mid-time step (00:30) as it
                ## is what NEMO assumes according to documentation
    
                ds = ds.interp(time=Time + dt2)
                ds.to_netcdf(foutInterp)
                ds.close()
    
            else:
                # under construction
                # nco is faster than python due to fortran under the hood
                # currently missing last record of year
    
                # merge all years
                command = "cdo mergetime {1}/{0}_y*.nc {1}/{0}_all.nc".format(
                                                         nameVar, self.path_EXTRACT)
                os.system(command)
    
                # interpolate
                finput = "{1}/{0}_all.nc".format(nameVar, self.path_EXTRACT)
                xrds = xr.open_dataset(finput, chunks=self.chunks)
                interp_time(xrds, finput, foutInterp)

    def interpolate_by_year(self, nameVar):
        for iY in range(self.year_init, self.year_end+1) :
            print (iY)

            path = self.path_EXTRACT + '/' + nameVar + '_y' 
            f0 = path + str(iY) + '.nc'
            ds0 = xr.open_dataarray(f0, chunks=self.chunks)
            if iY != self.year_end+1:
                f1 = path + str(iY+1) + '.nc'
                ds1 = xr.open_dataarray(f1, chunks=self.chunks)
                ds1 = ds1.isel(time=0)
                ds = xr.concat([ds0,ds1], dim='time')
            else:
                ds = ds0

            Time = ds.time.values
            dt = (Time[1] - Time[0]) / 2
            half_time = (ds.time + dt).sel(time=str(iY)).values
            ds = ds.interp(time=half_time)

            # format indexes and coords
            ds = self.format_nc(ds, nameVar)

            # save
            fout = self.path_FORCING + '/ERA5_' + nameVar + '_y' + str(iY) +'.nc'
            ds.to_netcdf(fout)

    def format_nc(self, da, nameVar):

        # mesh lat and lon
        mlon, mlat = np.meshgrid(da.longitude, da.latitude)
        lon_attrs={'long_name':'longitude','units':'degrees_east'}
        lat_attrs={'long_name':'latitude', 'units':'degrees_north'}
        mlon = xr.DataArray(mlon, dims=['Y','X'], attrs=lon_attrs)
        mlat = xr.DataArray(mlat, dims=['Y','X'], attrs=lat_attrs)
      
        # assign X/Y as indexes
        da = da.drop(['longitude','latitude'])
        da = da.rename({'longitude':'X','latitude':'Y'})
        da = da.assign_coords({'longitude':mlon,'latiude':mlat})
      
        # format fill values
        da = da.fillna(-9999999)
        da.attrs['_FillValue'] = -9999999
 
        return da
      

    def split_by_year(self, ds, outpath, var):
        for ind, year in ds_all.groupby('time.year'):
            print (ind)
            var = var.upper()
            year = self.cf_to_int_time(year)
            if nameVar in [ "d2m", "sp" ] :
                fout = "{2}/SPH_ERA5_{0}_y{1}.nc".format(var, ind, outpath)
            else:
                fout = "{2}/ERA5_{0}_y{1}.nc".format(var, ind, outpath)
            if clean : os.system( "rm {0}".format( fout ) )
            if not os.path.exists( fout ) :
                year.to_netcdf(fout)

    def process_all(self, step1=True, step2=True, step3=True):
        os.system( "mkdir {0} {1}".format( self.path_EXTRACT, self.path_FORCING ) )
        if self.west < 0 : self.west = 360.+self.west
        if self.east < 0 : self.east = 360.+self.east
        
        ## Loop over each variable
        for dirVar, nameVar in self.var_path.items() :
        
            print ("================== {0} - {1} ==================".format(
                    dirVar, nameVar ))
        
            ## -----------------------------------
            ## -------- step 1: EXTRACT ---------- 
            ## -----------------------------------
            if step1: self.extract_cut_out(nameVar)
        
            ## -----------------------------------
            #### ------ step 2: INTERPOLATE ------ 
            ## -----------------------------------
            if step2:
                #foutInterp = "{1}/{0}_all_interpt.nc".format(
                #nameVar, self.path_EXTRACT)
                ##@self.timeit
                #self.interpolate_all(nameVar, foutInterp, pythonic=self.pythonic)
                self.interpolate_by_year(nameVar)
        
#            ## -----------------------------------
#            ## ------ step 3: SPLIT BY YEAR ------ 
#            ## -----------------------------------
#
#            if step3:
#                # load interpolated
#                chunks={'time':5}
#                ds_all = xr.open_dataset(foutInterp, chunks=chunks)
#        
#                ds_all = self.set_attrs(ds_all)
#        
#                self.split_by_year(ds, self.path_FORCING, nameVar)
#            
        ##---------- PROCESS SPECIFIC HUMIDITY ----------------------     
        ## Compute Specific Humidity according to ECMWF documentation
        
        #if self.sph_ON : 
        #
        #    for iY in range( Year_init, Year_end+1 ) :
        #
        #        # Read
        #        d2m = self.read_NetCDF(
        #        "{1}/SPH_ERA5_D2M_y{0}.nc".format( iY, path_FORCING ), 'D2M',
        #        chunks=self.chunks)
        #        sp  = self.read_NetCDF( 
        #        "{1}/SPH_ERA5_SP_y{0}.nc" .format( iY, path_FORCING ), 'SP',
        #        chunks=self.chunks)
        #
        #        # Calculate sph
        #        esat = 611.21 * np.exp( 17.502 * (d2m-273.16) / (d2m-32.19) )
        #        dyrvap = 287.0597 / 461.5250
        #        dVar = dyrvap * esat / ( sp - (1-dyrvap) * esat)
        #        dvar.attrs = {'units':'1', 'standard_name':'specific humidity'}
        # 
        #        # Save
        #        Fout = "./{1}/ERA5_SPH_y{0}.nc".format( iY, path_FORCING )
        #        dVar.to_netcdf(Fout)

if __name__ == '__main__':
    era = era5()
    era.process_all(step1=False, step3=False)
