import os

## config.py ##

# 24/05/2016 R. D. Patmore
# configure and set paths

global plot_number; plot_number = 0
def readPath(case):
    readPath = '/gws/nopw/j04/bas_pog/ryapat30/' + case + '/run/'
    return readPath
def basgws():
    return '/gws/nopw/j04/bas_pog/ryapat30/'
def ISOBLcalc_path():
    return '/home/users/ryapat30/ISOBLcalc/'
def nemoPath():
    return '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/'
def nemoPath_coords():
    return '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/domain/'
def nemoPath_data():
    return '/gws/nopw/j04/nemo_vol1/ORCA0083-N006/means/'
def data_path():
    return '/gws/nopw/j04/nemo_vol1/ryapat30/SOCHIC/'
def root():
    return '/gws/nopw/j04/nemo_vol1/ryapat30/SOCHIC/'
