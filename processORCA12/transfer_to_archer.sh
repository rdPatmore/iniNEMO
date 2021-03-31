MODEL=EXP003
ARCH_PATH='/work/n02/n02/ryapat30/nemo/nemo/cfgs/SOCHIC_PATCH_ICE/'${MODEL}
DATE='y2015m11'

scp DataOut/restart_ice_conform.nc archer2:${ARCH_PATH}/data/restart_ice.nc
scp DataOut/restart_conform.nc archer2:${ARCH_PATH}/data/restart.nc
