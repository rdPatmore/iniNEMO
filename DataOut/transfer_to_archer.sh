MODEL=EXP08
ARCH_PATH='/work/n02/n02/ryapat30/nemo/nemoHEAD/cfgs/SOCHIC_ICE/'${MODEL}
DATE='y2014'
RES='24'

#scp coordinates.bdy.nc ${ARCH_PATH}
for pt in {U,V,T}; do
  cp -v ORCA${RES}/bdy_${pt}_ring_${DATE}.nc \
     ${ARCH_PATH}/data/bdy_${pt}_ring_${DATE}.nc
done

cp -v ORCA${RES}/bdy_I_ring_${DATE}_kelvin.nc \
    ${ARCH_PATH}/data/bdy_I_ring_${DATE}.nc
