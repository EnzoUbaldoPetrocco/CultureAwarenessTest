__author__ = "Enzo Ubaldo Petrocco"
import sys
sys.path.insert(1, "../../")


import os
import pathlib
import cv2
import random
import time
import tensorflow as tf
import numpy as np
from cleverhans.tf2.utils import optimize_linear
from matplotlib import pyplot as plt
from Utils.FileManager.FileManager import FileManagerClass

class Preprocessing:
    def create_ds(self, img_path, svpath, size):
        
        # create dir
        fileobj = FileManagerClass(svpath)
        # get images from root
        types = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for typ in types:
            paths.extend(pathlib.Path(img_path).glob(typ))
        for i, pt in enumerate(paths):
            im = cv2.imread(str(pt)) 
            #print(np.shape(im))
            # get resize them
            im = cv2.resize(im, (size, size), 
               interpolation = cv2.INTER_CUBIC)
            # save them with label
            cv2.imwrite(svpath + f"im{i}.jpg", im)

def main():
    prep = Preprocessing()
    basePt = "../../../../FINALDS/"
    lampsize = 120
    carpetsize = 200 

    chinoff = basePt + "originals/lamps/chinese/off/"
    chinon = basePt + "originals/lamps/chinese/on/"
    frenchoff = basePt + "originals/lamps/french/off/"
    frenchon = basePt + "originals/lamps/french/on/"
    turkoff = basePt + "originals/lamps/turkish/off/"
    turkon = basePt + "originals/lamps/turkish/on/"
    indoff = basePt + "originals/carpets/indian/without/"    
    indon = basePt + "originals/carpets/indian/with/"
    japoff = basePt + "originals/carpets/japanese/without/"
    japon = basePt + "originals/carpets/japanese/with/"
    scanoff = basePt + "originals/carpets/scandinavian/without/"
    scanon = basePt + "originals/carpets/scandinavian/with/"

    svchinoff = basePt + "/lamps/chinese/"
    svchinon = basePt + "/lamps/chinese/"
    svfrenchoff = basePt + "/lamps/french/"
    svfrenchon = basePt + "/lamps/french/"
    svturkoff = basePt + "/lamps/turkish/"
    svturkon = basePt + "/lamps/turkish/"
    svindoff = basePt + "/carpets_stretched/indian/"    
    svindon = basePt + "/carpets_stretched/indian/"
    svjapoff = basePt + "/carpets_stretched/japanese/"
    svjapon = basePt + "/carpets_stretched/japanese/"
    svscanoff = basePt + "/carpets_stretched/scandinavian/"
    svscanon = basePt + "/carpets_stretched/scandinavian/"


    prep.create_ds(chinoff, svchinoff + f"{lampsize}/RGB/off/", lampsize)
    prep.create_ds(chinon, svchinon + f"{lampsize}/RGB/on/", lampsize)
    prep.create_ds(frenchoff, svfrenchoff + f"{lampsize}/RGB/off/", lampsize)
    prep.create_ds(frenchon, svfrenchon + f"{lampsize}/RGB/on/", lampsize)
    prep.create_ds(turkoff, svturkoff + f"{lampsize}/RGB/off/", lampsize)
    prep.create_ds(turkon, svturkon + f"{lampsize}/RGB/on/", lampsize)
    prep.create_ds(indoff, svindoff + f"{carpetsize}/RGB/without/", carpetsize)
    prep.create_ds(indon, svindon + f"{carpetsize}/RGB/with/", carpetsize)
    prep.create_ds(japoff, svjapoff + f"{carpetsize}/RGB/without/", carpetsize)
    prep.create_ds(japon, svjapon + f"{carpetsize}/RGB/with/", carpetsize)
    prep.create_ds(scanoff, svscanoff + f"{carpetsize}/RGB/without/", carpetsize)
    prep.create_ds(scanon, svscanon + f"{carpetsize}/RGB/with/", carpetsize)


if __name__ == "__main__":
    main()
            