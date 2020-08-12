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


S = [0,1,2,3,26,4,5,6,7,8,26,26,26,26,26,9,26,10,26,26,26,11,12,13,26,14,15]
inter = codecs.open("inter.dat","r","utf-8")
dataset_trainmodif = codecs.open("dataset_trainmodif.csv","w","utf-8")
for line in inter.readlines():
    origin = line.split(",")
    fID = origin[-1].split("/")
    path = os.path.join(fID[4],origin[0]+'.jpg')
    print(path)
    image = io.imread(path)
    if image.ndim ==3 and image.shape[2] == 3 and int(origin[1])==1 and S[int(origin[2])]!=26:
        dataset_trainmodif.write(origin[0]+","+origin[1]+","+str(S[int(origin[2])])+","+origin[3]+"\n")
print('done')
dataset_trainmodif.close()
inter.close()

inter = codecs.open("inter.dat","r","utf-8")
dataset_testmodif = codecs.open("dataset_testmodif.csv","w","utf-8")
for line in inter.readlines():
    origin = line.split(",")
    fID = origin[-1].split("/")
    path = os.path.join(fID[4],origin[0]+'.jpg')
    print(path)
    image = io.imread(path)
    
    if image.ndim ==3 and image.shape[2] == 3 and int(origin[1])==2 and S[int(origin[2])]!=26:
        dataset_testmodif.write(origin[0]+","+origin[1]+","+str(S[int(origin[2])])+","+origin[3]+"\n")
print('done')
dataset_testmodif.close()
inter.close()


