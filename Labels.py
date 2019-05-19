# -*- coding: utf-8 -*-
"""

@author: ramon, bojana
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1145961, 1358297


def loadGT(fileName):
    """@brief   Loads the file with groundtruth content
    
    @param  fileName  STRING    name of the file with groundtruth
    
    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )
        
    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one lsit of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """

    scores = []
    for i in range(len(description)):
        scores.append(similarityMetric(description[i], GT[i][1], options))

    return sum(scores)/len(description), scores


def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """

    if options == None:
        options = {}
    if not 'metric' in options:
        options['metric'] = 'basic'

    S = 0
    if options['metric'].lower() == 'basic'.lower():
        S = len(set(Est).intersection(GT)) / float(len(Est))
    elif options['metric'].lower() == 'other'.lower():
        S = len(set(Est).intersection(GT)) / float(max(len(Est), len(GT)))

    return S


def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names
    
    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling
    
    @return meaningful_colors  LIST    colors labels of centroids of kmeans object
    @return unique             LIST    indexes of centroids with the same color label
    """

##  remind to create composed labels if the probability of
##  the best color label is less than  options['single_thr']

    """
    indexes = np.argmax(kmeans.centroids, axis=1)
    meaningful_colors = np.unique([cn.colors[i] for i in indexes])
    unique=[]
    for color in meaningful_colors:
        unique.append([i for i, x in enumerate(index) if x == cn.colors.index(color)])
    """

    """
    centroids_index = np.zeros(len(kmeans.centroids))
    for i in range(len(kmeans.X)):
        centroids_index[int(kmeans.clusters[i])] += 1
    kmeans.centroids = kmeans.centroids[np.argsort(-centroids_index)]
    """

    meaningful_colors = []
    indexes = []
    unique = []
    probability = np.copy(kmeans.centroids)

    for i in range(len(kmeans.centroids)):
        index = np.argmax(probability[i])
        if options["single_thr"] >= probability[i][index]:
            # Compound Labels #
            next_index = np.argmax(np.delete(probability[i], index))
            if next_index >= index:
                next_index += 1
            indexes.append([index, next_index])
        else:
            # Simple Labels #
            indexes.append([index])

    for indexList in indexes:
        name = ""
        for i in indexList:
            if not name or ord(name[0]) <= ord(cn.colors[i][0]):
                name += cn.colors[i]
            else:
                name = cn.colors[i]+name
        meaningful_colors.append(name)
        indices = [i for i, x in enumerate(indexes) if x == indexes[indexes.index(indexList)]]
        unique.append(indices)

    return list(np.unique(meaningful_colors)), list(np.unique(unique))


def processImage(im, options):
    """@brief   Finds the colors present on the input image
    
    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options
    
    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################

##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
    if options['colorspace'].lower() == 'ColorNaming'.lower():
        im = cn.ImColorNamingTSELabDescriptor(im)
    elif options['colorspace'].lower() == 'RGB'.lower():
        pass 
    elif options['colorspace'].lower() == 'Lab'.lower():
        im = color.rgb2lab(im)

##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K'] < 1:    # find the best K #
        kmeans = km.KMeans(im, 0, options)
        kmeans.bestK()
    else:
        kmeans = km.KMeans(im, options['K'], options) 
        kmeans.run()

##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
    if options['colorspace'].lower() == 'RGB'.lower():
        kmeans.centroids = cn.ImColorNamingTSELab(kmeans.centroids.reshape((1, -1, 3)))[0]
        kmeans.centroids = np.reshape(kmeans.centroids, (kmeans.K, 11))
    elif options['colorspace'].lower() == 'Lab'.lower():
        kmeans.centroids = np.reshape(kmeans.centroids, (-1, 1, kmeans.centroids.shape[1]))
        kmeans.centroids = color.lab2rgb(kmeans.centroids)*255
        kmeans.centroids = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
        kmeans.centroids = np.reshape(kmeans.centroids, (-1, kmeans.centroids.shape[2]))

#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)
    return colors, which, kmeans