MODEL='SOCHIC_12'
MONTH=11
YEAR=2015

scp archer2:/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/${MODEL}/\
restart_ice_y${YEAR}m${MONTH}.nc DataIn/restart12_ice.nc
scp archer2:/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/${MODEL}/\
restart_y${YEAR}m${MONTH}.nc DataIn/restart12.nc
