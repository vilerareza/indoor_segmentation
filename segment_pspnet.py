# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 09:26:43 2022

@author: Reza Vilera
"""

import os
import shutil
import mxnet as mx
from mxnet import image 
from mxnet.gluon.data.vision import transforms
import gluoncv
from matplotlib import pyplot as plt
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.data.transforms.image import imresize
from gluoncv.utils.viz import get_color_pallete
import numpy as np
import cv2 as cv

# Preparing output dirs
def create_output_dir(base_dir = 'output', child_dirs = ['wall', 'floor', 'table', 'wardrobe', 'kitchen']):
    '''Preparing output directory'''
    # Create directory named 'output'. Overwrite if already exist
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir (base_dir)
    # Create sub dirs based on child_dirs argument
    for name in child_dirs:
        os.mkdir(os.path.join(base_dir, name))
    return base_dir, child_dirs

# Getting input filenames
def get_input_filenames (input_dir = 'input'):
    try:
        fileNames = os.listdir(input_dir)
        return fileNames
    except:
        return []

# Segmentation prediction
def predict(model, input_dir = 'input', output_dir = 'output', objectDict = {'wall' : 1, 'floor' : 4}):
    
    masks = []
    fileNames = get_input_filenames(input_dir)
    outputClasses = os.listdir(output_dir)
    
    for fileName in fileNames:
        img = image.imread(os.path.join(input_dir, fileName))
        #imgNp_ori = img.asnumpy()
        img = imresize(img, 256, 256)
        imgNp = img.asnumpy()
        imgNp = imgNp[:,:,::-1]
        imgGray = cv.cvtColor(imgNp, cv.COLOR_RGB2GRAY)
        imgNp = cv.normalize(imgNp, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
        img = test_transform(img, ctx)
        
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        
        mask = get_color_pallete(predict, 'ade20k')
        mask = np.array(mask)
        
        for outputClass in outputClasses:
            # For eveery output class
            try:
                # Try to find the class id in objectDict based on the output directory names (outputClasses)
                classID = objectDict[outputClass]
                
                '''binary mask generation'''
                # Empty array for binary mask
                mask_binary = np.zeros([mask.shape[0], mask.shape[1]])
                # Iterate to every element in mask_binary. Set the element to 255 if the value on same element in mask is equal to class_id
                it = np.nditer(mask_binary, flags = ['multi_index'], op_flags = ['readwrite'])
                for pixel in it:
                    if (mask[it.multi_index[0],it.multi_index[1]]== classID):
                        pixel[...] = 255

                '''color blender generation'''
                blend = np.ones((256,256,3))
                blend = cv.normalize(blend, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
                
                it = np.nditer(mask_binary, flags = ['multi_index'], op_flags = ['readwrite'])
                for pixel in it:
                    if (pixel == 255):
                        blend[it.multi_index[0], it.multi_index[1],:] = [0,0.5,0]
                imgBlend = cv.addWeighted(imgNp,0.5, blend, 0.5,0)
                    
                cv.imwrite(os.path.join(output_dir, outputClass, 'blend'+fileName), imgBlend*255)                                                               
                cv.imwrite(os.path.join(output_dir, outputClass, fileName), mask_binary)
            
            except Exception as e:
                print (e)

        
# classes
obj_classes = ['wall', 'floor', 'table', 'wardrobe', 'kitchen']
# Getting input filenames
fileNames = get_input_filenames()
# Creating output dirs
outDir, outputClasses = create_output_dir(child_dirs=obj_classes)
#objectDict = {'wall' : 1, 'floor' : 4, 'wardrobe': 36, 'cabinet': 11, 'table':16, 'shelf':25, 'desk': 34, 'coffee_table': 65, 'stove': 72, }
objectID = {'wall' : 1, 'floor' : 4, 'kitchen': 74, 'wardrobe': 36, 'table': 11}
# Predict
ctx = mx.cpu(0)
model = gluoncv.model_zoo.get_model('psp_resnet50_ade', pretrained=True)
predict(model = model, input_dir = 'input', output_dir = 'output', objectDict = objectID)