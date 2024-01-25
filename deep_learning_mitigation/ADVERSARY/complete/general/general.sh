#!/bin/sh

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 0 "carpets" "c_ind.csv" 10 0.05
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 0 "carpets" "c_ind.csv" 10 0.1
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 1 "carpets" "c_jap.csv" 10 0.05
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 1 "carpets" "c_jap.csv" 10 0.1
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 2 "carpets" "c_scan.csv" 10 0.05
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 2 "carpets" "c_scan.csv" 10 0.1
done


for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 0 "lamps" "l_chin.csv" 10 0.05
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 0 "lamps" "l_chin.csv" 10 0.1
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 1 "lamps" "l_fren.csv" 10 0.05
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 1 "lamps" "l_fren.csv" 10 0.1
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 2 "lamps" "l_tur.csv" 10 0.05
done

for c in -1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
do 
python3 general.py 2 "lamps" "l_tur.csv" 10 0.1
done

