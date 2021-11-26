MODEL=EXP08
ARCH_PATH='/work/n02/n02/ryapat30/nemo/nemoHEAD/cfgs/SOCHIC_ICE/'${MODEL}

#scp ECMWF_24_conform.nc \
#    archer2:${ARCH_PATH}/data/ECMWF_24_y2015m01.nc
#scp ECMWF_03_conform.nc \
#    archer2:${ARCH_PATH}/data/ECMWF_03_y2015m01.nc

cp -v ORCA24/DFS5.2_03_y2014.nc \
    ${ARCH_PATH}/data/DFS5.2_03_y2013.nc
cp -v ORCA24/DFS5.2_24_y2014.nc \
    ${ARCH_PATH}/data/DFS5.2_24_y2013.nc

#scp sss_1m_conform.nc \
#    archer2:${ARCH_PATH}/data/
