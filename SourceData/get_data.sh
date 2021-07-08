#year=2012
#months='01 02 03 04 05 06 07 08 09 10 11 12'
#days='01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31'

year=2014
months='01 02 03 04 05 06 07 08 09 10 11 12'
days='01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31'

#year=2012
#months='10'
#days='01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31'
for pt in {U,V,T,I}; do
  for month in $months; do
    for day in $days; do
      wget http://gws-access.jasmin.ac.uk/public/nemo/runs/ORCA0083-N06/means/${year}/ORCA0083-N06_${year}${month}${day}d05${pt}.nc
    done
  done
done
