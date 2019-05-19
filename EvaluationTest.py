# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
import matplotlib.pyplot as plt
import os
import time 
import numpy as np

if os.path.isfile('TeachersLabels.py') and True: 
    import TeachersLabels as lb
else:
    import Labels as lb



plt.close("all")
if __name__ == "__main__":
    max_iter=1
    
    print("max_iter",max_iter)

    #'colorspace': 'RGB', 'Lab' o 'ColorNaming'
    
    encert_promig=np.zeros(8)
    t = time.time()
    idx=0
    while (idx< max_iter):
        
        for i in range(3,11):
            
            options = {'colorspace':'ColorNaming', 'K':i, 'synonyms':False, 'single_thr':0.2, 'verbose':False, 'km_init':'random', 'metric':'basic'}
            #print(i)
            ImageFolder = 'Images'
            GTFile = 'LABELSlarge.txt'
            GTFile = ImageFolder + '/' + GTFile
            GT = lb.loadGT(GTFile)
            DBcolors = []
            for gt in GT:
                
                im = io.imread(ImageFolder+"/"+gt[0])    
                colors,_,_ = lb.processImage(im, options)
                DBcolors.append(colors)
            encert,_ = lb.evaluate(DBcolors, GT, options)
            encert_promig[i-3]+=encert
           
        idx+=1
  
    encert_promig=(encert_promig/max_iter)*100   
    print((encert_promig[0])) 
    print(time.time()-t)
    #print(encert_promig)
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    im = io.imread(ImageFolder+"/"+gt[0])  
    plt.figure(1)
    plt.imshow(im)
    plt.axis('off')
    K=np.array(np.arange(3,11))    
    plt.figure(2)
    plt.cla()
    plt.plot(K,encert_promig,label='encert_promig %')
    plt.xlabel('K')
    plt.ylabel('% encert')
    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(0.01)
