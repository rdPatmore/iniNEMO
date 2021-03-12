ARCH_PATH='/work/n02/n02/ryapat30/nemo/Data/ORCA12/'
year=2015
month=01
day=10

for pt in {T,U,V}; do
  scp DataOut/ORCA0083-N06_y${year}m${month}_${pt}_conform.nc \
        archer2:${ARCH_PATH}
  scp DataOut/ORCA0083-N06_y${year}m${month}d${day}_${pt}_conform.nc \
        archer2:${ARCH_PATH}
done

scp DataOut/coordinates_subset.nc archer2:${ARCH_PATH}
