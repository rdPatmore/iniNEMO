MODEL=EXP106
ARCH_PATH='/work/n02/n02/ryapat30/nemo/nemo/cfgs/SOCHIC_PATCH_B/'${MODEL}
DATE='y2015m01'

scp coordinates.bdy.nc archer2:${ARCH_PATH}
for pt in {U,V,T}; do
  scp BdyOut/bdy_${pt}_ring_${DATE}.nc \
      archer2:${ARCH_PATH}/data/bdy_${pt}_ring_${DATE}.nc
done
