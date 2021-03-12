ARCH_PATH='/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_24'

for pt in {T,U,V}; do
  for bound in {east,west,north,south}; do
    scp archer2:${ARCH_PATH}/bdy_${pt}_${bound}_y2015m01d03.nc \
    bdy_${pt}_${bound}_y2015m01.nc
  done
done
#scp archer2:/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_IN/bdy_U_east_y2015m01d03.nc bdy_U_east.nc
#scp archer2:/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_IN/bdy_V_east_y2015m01d03.nc bdy_V_east.nc
#scp archer2:/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_IN/bdy_T_east_y2015m01d03.nc bdy_T_east.nc
