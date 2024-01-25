conda activate tensorflow
ECHO Jan
for /l %x in (0, 1, 25) do python3 general.py 0 "carpets" "c_ind.csv" 10 0.05
ECHO jirijan
PAUSE