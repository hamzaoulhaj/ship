from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import traceback
import threading
import datetime
import logging
import codecs
import math
import sys
import os

downloadFile = codecs.open("VesselClassification.dat","r","utf-8")
downloadContent = downloadFile.readlines()
downloadFile.close()

allPaths = []
allIDs = []
dirs = os.listdir(os.getcwd())
for eachDir in dirs:
    if 'W' in eachDir:
        FinalList = os.listdir(os.path.join(os.getcwd(),eachDir))
        for eachFile in FinalList:
            if ".jpg" in eachFile:
                fPath = os.path.join(os.getcwd(),eachDir,eachFile)
                fID = eachFile.split(".")[0]
                allPaths.append(fPath)
                allIDs.append(fID)



inter = codecs.open("inter.dat","w","utf-8")
for eachLine in downloadContent:
    tempID = eachLine.split(",")[0]
    try:
        tempIndex = allIDs.index(tempID)
        label = eachLine.split(",")[2]
        train = eachLine.split(",")[1]
        inter.write(tempID+","+train+","+label+","+str(allPaths[tempIndex])+"\n")
    except:
        continue
inter.close()


inter = codecs.open("inter.dat","r","utf-8")
dataset_train = codecs.open("dataset_train.csv","w","utf-8")
for line in inter.readlines():
    origin = line.split(",")
    fID = origin[-1].split("/")
    path = os.path.join(fID[4],origin[0]+'.jpg')
    print(path)
    image = io.imread(path)
    if image.ndim ==3 and image.shape[2] == 3 and int(origin[1])==1:
        dataset_train.write(line)

dataset_train.close()
inter.close()

inter = codecs.open("inter.dat","r","utf-8")
dataset_test = codecs.open("dataset_test.csv","w","utf-8")
for line in inter.readlines():
    origin = line.split(",")
    fID = origin[-1].split("/")
    path = os.path.join(fID[4],origin[0]+'.jpg')
    image = io.imread(path)
    
    if image.ndim ==3 and image.shape[2] == 3 and int(origin[1])==2:
        dataset_test.write(line)

dataset_test.close()
inter.close()

