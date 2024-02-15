import csv
import os


class FileManagerClass:
    def __init__(self, name):
        self.name = name
        dir = os.path.dirname(name)
        self.mkdir(dir)

    def mkdir(self, dir):
        try:
            if not os.path.exists(dir):
                print(f"Making directory: {str(dir)}")
                os.makedirs(dir)
        except Exception as e:
            print(f"{dir} Not created")

    def readrows(self):
        csvlist = []
        try:
            with open(self.name, "r") as file:
                csvreader = csv.reader(file)
                for row in csvreader:
                    csvlist.append(row)                
                file.close()
        except:
            print(f"Error in reading file {self.name}")
        return csvlist

    def writerow(self, row):
        try:
            with open(self.name, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)
                del row
                file.close()
        except Exception as e:
            print(f"Error in writing file {self.name} due to Exception:\n{e}")

    def writecm(self, cm):
        row = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        self.writerow(row)
        del row

    def readcms(self):
        cms = []
        rows = self.readrows()
        for row in rows:
            cm = [[int(row[0]), int(row[1])], [int(row[2]), int(row[3])]]
            cms.append(cm)
            del cm
        return cms

