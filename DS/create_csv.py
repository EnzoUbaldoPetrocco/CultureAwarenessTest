import sys
sys.path.insert(1, '../')
from Utils import utils
import csv
import os


os.remove('LAMP.csv')
os.remove('CARPET.csv')
# Create csv for LAMP dataset
first_row = ['Culture', 'Image', 'Class']
try:
    with open('LAMP.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
except Exception as e:
    print(f'Error in writing file LAMP.CSV due to Exception:\n{e}')
# Create csv for CARPET dataset
try:
    with open('CARPET.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
except Exception as e:
    print(f'Error in writing file LAMP.CSV due to Exception:\n{e}')

