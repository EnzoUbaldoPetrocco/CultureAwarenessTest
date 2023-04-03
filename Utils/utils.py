import csv
import numpy as np

class FileClass:
    def __init__(self, name):
        self.name = name

    def readrows(self):
        csvlist = []
        try:
            with open(self.name, 'r') as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    csvlist.append(row)
        except:
            print(f'Error in reading file {self.name}')
        return csvlist
    
    def writerow(self, row):
        try:
            with open(self.name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
        except:
            print(f'Error in writing file {self.name}')

    def writecm(self, cm):
        row = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        self.writerow(row)

    def readcms(self):
        cms = []
        rows = self.readrows()
        for row in rows:
            cm = [[row[0], row[1]], [row[2], row[3]]]
            cms.append(cm)
        return cms