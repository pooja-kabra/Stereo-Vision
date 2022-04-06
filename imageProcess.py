# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 20:58:50 2022

@author: pooja
"""
import numpy as np
import cv2 as cv

def displayAndSaveImage(image, file_name =  None):
    cv.imshow("image", image)
    cv.waitKey(0)
    if file_name is not None:
        cv.imwrite(file_name, image)

def normalize(image):
    maxval = image.max()  
    minval = image.min()  
    span = maxval - minval
    
    image = (image - minval)/span
    image = image*255         
    image = image.astype(np.uint8)
    
    return image