#! /usr/bin/env python3

import os
import zipfile
import pathlib
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv

file_name = "../accese vs spente.zip"
# opening the zip file in READ mode
with zipfile.ZipFile(file_name, 'r') as zip:
    zip.extractall('../')
    print('Done!')


chinese = []
chinese_categories = []
french = []
french_categories = []

path = '../chinese'
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)
path = '../french'
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

def get_dimensions(height, width):
  list_size = []
  list_size.append(math.floor((700 - height)/2))
  list_size.append(math.ceil((700 - height)/2))
  list_size.append(math.floor((500 - width)/2))
  list_size.append(math.ceil((500 - width)/2))
  return list_size

def manage_size(im):
  dimensions = im.shape
  while dimensions[0]>700 or dimensions[1]>500:
    im = cv2.resize(im,(int(dimensions[0]*0.5),int(dimensions[1]*0.5)),interpolation = cv2.INTER_AREA )
    dimensions = im.shape
  return im

def fill_chinese():
  global chinese, chinese_categories
  path = '../accese vs spente/cinesi/'
  #paths_chin_off = pathlib.Path(path).glob('*.png')
  types = ('*.png', '*.jpg', '*.jpeg') # the tuple of file types
  paths_chin_off = []
  for files in types:
      paths_chin_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_off = sorted([x for x in paths_chin_off])

  for i in ds_sorted_chin_off:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])

    im = rgb2gray(im)

    im_obj = pd.DataFrame(im)

    chinese.append(im_obj)
    #chinese.append(im.flatten())
    chinese_categories.append(0)
  path = '../accese vs spente/cinesi accese/'
  paths_chin_on = []
  for files in types:
      paths_chin_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_chin_on = sorted([x for x in paths_chin_on])
  for i in ds_sorted_chin_on:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)

    im_obj = pd.DataFrame(im)

    chinese.append(im_obj)
    chinese_categories.append(1)
  return chinese


def fill_french():
  global french, french_categories
  path = '../accese vs spente/francesi accese/'
  types = ('*.png', '*.jpg', '*.jpeg')
  paths_fren_on = []
  for files in types:
      paths_fren_on.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_on = sorted([x for x in paths_fren_on])
  for i in ds_sorted_fren_on:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)
    im_obj = pd.DataFrame(im)
    french.append(im_obj)
    french_categories.append(1)
  path = '../accese vs spente/francesi/'
  paths_fren_off = []
  for files in types:
      paths_fren_off.extend(pathlib.Path(path).glob(files))
  ds_sorted_fren_off = sorted([x for x in paths_fren_off])
  for i in ds_sorted_fren_off:
    im = cv2.imread(str(i))
    im = manage_size(im)
    dimensions = im.shape
    tblr = get_dimensions(dimensions[0],dimensions[1])
    im = cv2.copyMakeBorder(im, tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=[255,255,255])
    im = rgb2gray(im)
    im_obj = pd.DataFrame(im)

    french.append(im_obj)
    french_categories.append(0)
  return french


fill_chinese()
fill_french()

with open('../chinese/chinese.csv', 'w') as csvfile:
  for i in range(0, len(chinese)):
# open the file in the write mode
    fieldnames = ['image', 'category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'image': chinese[i], 'category': chinese_categories[i]})


with open('../french/french.csv', 'w') as csvfile:
  for i in range(0, len(french)):
# open the file in the write mode
    fieldnames = ['image', 'category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'image': french[i], 'category': french_categories[i]})


