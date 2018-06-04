import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shutil import copyfile


# Import The Dataset
dataset = pd.read_csv('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\test\\img_ids.txt' , header=None)
x = dataset.iloc[:, 0:1].values
l=[]
for i in range(len(x[:,0])):
    if x[i,0] not in l:
        l.append(x[i,0])


tmp = ['000000000000',
       '00000000000',
       '0000000000',
       '000000000',
       '00000000',
       '0000000',
       '000000',
       '00000',
       '0000',
       '000',
       '00',
       '0',
       ''
       ]
src = "C:\\Users\\abdullahfcis\\Desktop\\Dataset\\val2014\\COCO_val2014_"   
dst = "C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\test\\Images\\COCO_val2014_"   
for i in range(len(l)):
   copyfile(src + tmp[len(str(l[i]))] + str(l[i]) + ".jpg" , dst + tmp[len(str(l[i]))] + str(l[i]) + ".jpg")
