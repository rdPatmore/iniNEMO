MODEL=EXP50
ARCH_PATH='/work/n02/n02/ryapat30/nemo/nemo/cfgs/SOCHIC_PATCH_B/'${MODEL}

scp coordinates.bdy.nc archer2:${ARCH_PATH}

for pt in {U,V,T}; do
  scp BdyOut/bdy_${pt}_ring.nc \
      archer2:${ARCH_PATH}/data/bdy_${pt}_ring_y2015m01.nc
done
