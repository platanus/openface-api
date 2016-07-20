#!/usr/bin/env python2
#
# Platanus 2016
# Based on:
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
import cv2
import itertools
import os
import urllib
import numpy as np
import openface

#init            
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
dlibFacePredictor = "shape_predictor_68_face_landmarks.dat"
openfaceModelDir = os.path.join(modelDir, 'openface')
align = openface.AlignDlib(dlibFacePredictor)
networkModel = "nn4.small2.v1.t7"
imgDim = 96
net = openface.TorchNeuralNet(networkModel, imgDim)
np.set_printoptions(precision=2)

def perform(img1, img2):
    dist = getRep(img1) - getRep(img2)
    return np.dot(dist, dist)

def getRep(imgPath):    
    req = urllib.urlopen(imgPath)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    bgrImg = cv2.imdecode(arr,-1) # 'load it as it is'    
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))        
    alignedFace = align.align(imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))        
    rep = net.forward(alignedFace)    
    return rep