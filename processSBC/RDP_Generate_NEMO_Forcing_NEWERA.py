#
#====================== DOCSTRING ============================
"""
Generate ERA5 atmospheric forcing for NEMO
So far to prduce a year by a year - need to be automated
Loosly based on code by Nico
--------------------------------------------------------------
"""
__author__      = "Nicolas Bruneau and Ryan Patmore"
__copyright__   = "Copyright 2018, NOC"
__email__       = "ryapat@noc.ac.uk"
__date__        = "2023-05"

#====================== USR PARAMS ===========================

Year_init    = 1989                                ## First year to process
Year_end     = 1990                                ## Last one [included]
East         =   19                                             ## East Border
West         =  -28                                             ## West Border
North        =   68                                             ## North Border
South        =   38                                             ## South Border
path_ERA5    = '/projectsa/NEMO/Forcing/ERA5/SURFACE_FORCING' 
## ROOT PATH OF ERA5 DATA
path_EXTRACT = '/projectsa/NEMO/ryapat/Extract' ## WHERE TO EXTRACT YOUR REGION
path_FORCING = '/projectsa/NEMO/ryapat/Forcing' ## NEMO FORCING
clean        = False                        ## Clean extraction (longest bit)
sph_ON       = True                         ## Compute specific humidity or not

#================== NEMO DOCUMENTATION =======================

"""
See the manual in section SBC for more details on the way data
are used in NEMO
The time variable from the netcdf is not used
"""

#====================== LOAD MODULES =========================

import os, sys, glob
import numpy as np
import datetime
from   netCDF4 import Dataset, MFDataset
import xarray as xr

#====================== VARIABLE DEF =========================

var_path = { "10m_u_component_of_wind" : "u10", \
             "10m_v_component_of_wind" : "v10", \
             "2m_temperature"          : "t2m", \
             "mean_sea_level_pressure" : "msl", \
             "mean_snowfall_rate"      : "msr" , \
             "mean_surface_downward_long_wave_radiation_flux"  : "msdwlwrf", \
             "mean_surface_downward_short_wave_radiation_flux" : "msdwswrf", \
             "mean_total_precipitation_rate" : "mtpr" }

if sph_ON :
   var_path[ "surface_pressure"  ] = 'sp'
   var_path[ "2m_dewpoint_temperature" ] = 'd2m'

#===================== INTERNAL FCTNS ========================

def Read_NetCDF( fname, KeyVar ) :
    """Read NetCDF file"""

    if "*" in fname: 
        lfiles = sorted( glob.glob( fname ) )
        ds = xr.open_mfdataset(lfiles, chunks={'longitude':100,'latitude': 100},
                                       parallel=True)
    else: 
        ds = xr.open_dataset(fname, chunks={'longitude':100,'latitude': 100},
                                       parallel=True)

    try:
        Lon = ds.longitude
        Lat = ds.latitude
        LON, LAT = np.meshgrid( Lon, Lat )
    except:
        LON = ds.lon
        LAT = ds.lat
    out = ds[KeyVar]
    tout = ds[KeyVar].time.values
    
    print (tout[0], tout[-1], tout.shape, out.shape, LON.shape)
    try    : return tout, LON, LAT, out, out.units, out.long_name
    except : return tout, LON, LAT, out, out.units, out.standard_name

#======================= EXTRACTION ==========================

def Extract( fin, fout, clean=True ) :
    if clean : os.system( "rm {0}".format( fout ) )
    if not os.path.exists( fout ) :
       command = "ncks -d latitude,{0},{1} -d longitude,{2},{3} {4} {5}".format(
                  np.float(South), np.float(North),
                  np.float(West), np.float(East), fin, fout )
       print (command)
       os.system( command )
      
def datetime_range(start, end, delta):
    current = Fstart, ]
    while current[-1] < end:
        current.append( current[-1]+delta )
    return np.array(current)

#======================= CORE PROGR ==========================
## load NCO
os.system( "module load nco/gcc/4.4.2.ncwa" )
os.system( "mkdir {0} {1}".format( path_EXTRACT, path_FORCING ) )
if West < 0 : West = 360.+West
if East < 0 : East = 360.+East

## Loop over each variable
for dirVar, nameVar in var_path.items() :

    print ("================== {0} - {1} ==================".format( dirVar, nameVar ))

    ##---------- EXTRACT ALL DATA FOR DOMAIN ----------------
    for iY in range( Year_init, Year_end+1 ) :
        ## Files
        finput  = "{0}/{1}/{2}_{1}.nc".format( path_ERA5, dirVar, iY )
        foutput = "{2}/{0}_{1}.nc".format( nameVar, iY, path_EXTRACT )
        ## Extract the subdomain
        Extract( finput, foutput, clean=clean ) 

    ##---------- LOAD FULLL TIME SERIES IN MEMORY -----------
    Time, Lon, Lat, dum, Units, Name = Read_NetCDF(
                       "{1}/{0}_*.nc".format( nameVar, path_EXTRACT ), nameVar )
    print ("Time" , Time)

    ## assume to be constant in time
    dt  = (Time[1] - Time[0]).astype('timedelta64[s]') 
    dt2 = dt / 2
    print ("dt", dt, dt2)

    ##---------- SOME PREPROCESSING -------------------------
    ## Add time step for last hour - copy the last input
    ## instantaneous field every hour. we center 
    ## it in mid-time step (00:30) as it
    ## is what NEMO assumes according to documentation

    dumC = dum.interp(time=dum.time.values + dt2)
    TimeC = dumC.time
    suffix = ''

    print ("TimeC", TimeC)

    ##---------- OUTPUT A FILE PER YEAR ---------------------
    for ind, year in dumC.groupby('time.year'):
        if nameVar in [ "d2m", "sp" ] :
            Fout = "{2}/forSPH_ERA5_{0}_y{1}.nc".format(
                                             nameVar.upper(), ind, path_FORCING)
        else : 
            Fout = "{2}/ERA5_{0}_y{1}.nc".format(
                                             nameVar.upper(), ind, path_FORCING)
        try:
            year.to_netcdf(Fout, mode='a')
        except:
            year.to_netcdf(Fout)

##---------- PROCESS SPECIFIC HUMIDITY ----------------------     
## Compute Specific Humidity according to ECMWF documentation

if sph_ON : 

   for iY in range( Year_init, Year_end+1 ) :
       Time, Lon, Lat, d2m, dUnits, dName = Read_NetCDF( "{1}/forSPH_ERA5_D2M_y{0}.nc".format( iY, path_FORCING ), 'D2M' )
       Time, Lon, Lat, sp , dUnits, dName = Read_NetCDF( "{1}/forSPH_ERA5_SP_y{0}.nc" .format( iY, path_FORCING ), 'SP'  )
       esat = 611.21 * np.exp( 17.502 * (d2m-273.16) / (d2m-32.19) )
       dyrvap = 287.0597 / 461.5250
       dVar = dyrvap * esat / ( sp - (1-dyrvap) * esat)
       Units = "1"; Name = "Specific Humidity"

       indT = ( Time >= datetime.datetime( iY  ,1,1 ) ) \
            * ( Time <  datetime.datetime( iY+1,1,1 ) )

       Fout = "./{1}/ERA5_SPH_y{0}.nc".format( iY, path_FORCING )
       try:
           year.to_netcdf(Fout, mode='a')
       except:
           year.to_netcdf(Fout)


    
