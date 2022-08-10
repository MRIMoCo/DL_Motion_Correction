#       !/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Permute
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K  
#from data import *
import keras.backend as K
import numpy as np
import os,glob
import re
import numpy as np
from PIL import Image as im
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
keras.backend.set_image_data_format('channels_first')

#IMG_HEIGHT, IMG_WIDTH = 128, 128
IMG_HEIGHT, IMG_WIDTH = 256, 256
SEED=42

img_height = IMG_HEIGHT
img_width  = IMG_WIDTH

def total_variation_loss(y, x):
    #xx_vect = x[:, :img_height - 1, :img_width - 1, :]
    #print(x.type())    
    #an_array = K.eval(x)
    #xx_vect = xx_vect.flatten()

    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])    
    #print('max of a ' , K.max(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :]), K.min(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :]))
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

    
from keras.layers import *
from keras.models import Model

def conv_bn_relu(x,num_filters=32,kernel_size=3):
    x = Conv2D(num_filters, kernel_size, padding = 'same',kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def conv_bn_relu_channels_first(x,num_filters=32,kernel_size=3):
    x = Conv2D(num_filters, kernel_size, padding = 'same',kernel_initializer = 'he_normal', data_format = 'channels_first')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def UNet_Total_Variation_BN_NCHW(IMG_HEIGHT, IMG_WIDTH):
    #Channel first
    inputs = Input((1, IMG_HEIGHT, IMG_WIDTH))
    print('image size ', IMG_HEIGHT, IMG_WIDTH)
    conv1 = conv_bn_relu_channels_first(inputs,32, 3)
    conv1 = conv_bn_relu_channels_first(conv1, 32, 3)
    print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn_relu_channels_first(pool1, 64, 3)
    conv2 = conv_bn_relu_channels_first(conv2, 64, 3)
    print(conv2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn_relu_channels_first(pool2,128, 3)
    conv3 = conv_bn_relu_channels_first(conv3, 128, 3)
    print(conv3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_bn_relu_channels_first(pool3, 256, 3)
    conv4 = conv_bn_relu_channels_first(conv4,256, 3)
    print('conv4 ', conv4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_bn_relu_channels_first(pool4, 512, 3)
    conv5 = conv_bn_relu_channels_first(conv5, 512, 3)
    print('conv5 ', conv5.shape)
    up6 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv5)),256, 3)
    #
    print('up 6 ', up6.shape)
    merge6 = concatenate([conv4,up6], axis = 1)
    print('merge 6 = conv4 + up6 ', merge6.shape)
    conv6 = conv_bn_relu_channels_first(merge6, 256, 3)
    conv6 = conv_bn_relu_channels_first(conv6, 256, 3)
    up7 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv6)), 128, 3)
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = conv_bn_relu_channels_first((merge7),128, 3)
    conv7 = conv_bn_relu_channels_first((conv7),128, 3)
    up8 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv7)),64, 3)
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = conv_bn_relu_channels_first((merge8),64, 3)
    conv8 = conv_bn_relu_channels_first((conv8),64, 3)
    up9 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv8)),32, 3)
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = conv_bn_relu_channels_first((merge9),32, 3)
    conv9 = conv_bn_relu_channels_first((conv9),32, 3)    
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid',data_format = 'channels_first', name = 'conv10')(conv9)
    convTV = Conv2D(1, 1, activation = 'sigmoid',data_format = 'channels_first', name = 'convTV')(conv9)
    #addInToOut = Add(name = 'addInToOut')([inputs, conv10])     
    model = Model(input = inputs, output = [conv10, convTV])
    return model

    
def UNet_Total_Variation_BN_InToOut_NCHW(IMG_HEIGHT, IMG_WIDTH):
    #Channel first
    inputs = Input((1, IMG_HEIGHT, IMG_WIDTH))
    print('image size ', IMG_HEIGHT, IMG_WIDTH)
    conv1 = conv_bn_relu_channels_first(inputs,32, 3)
    conv1 = conv_bn_relu_channels_first(conv1, 32, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_bn_relu_channels_first(pool1, 64, 3)
    conv2 = conv_bn_relu_channels_first(conv2, 64, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_bn_relu_channels_first(pool2,128, 3)
    conv3 = conv_bn_relu_channels_first(conv3, 128, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_bn_relu_channels_first(pool3, 256, 3)
    conv4 = conv_bn_relu_channels_first(conv4,256, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_bn_relu_channels_first(pool4, 512, 3)
    conv5 = conv_bn_relu_channels_first(conv5, 512, 3)
    up6 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv5)),256, 3)
    #
    merge6 = concatenate([conv4,up6], axis = 1)
    conv6 = conv_bn_relu_channels_first(merge6, 256, 3)
    conv6 = conv_bn_relu_channels_first(conv6, 256, 3)
    up7 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv6)), 128, 3)
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = conv_bn_relu_channels_first((merge7),128, 3)
    conv7 = conv_bn_relu_channels_first((conv7),128, 3)
    up8 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv7)),64, 3)
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = conv_bn_relu_channels_first((merge8),64, 3)
    conv8 = conv_bn_relu_channels_first((conv8),64, 3)
    up9 = conv_bn_relu_channels_first((UpSampling2D(size = (2,2), data_format = 'channels_first')(conv8)),32, 3)
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = conv_bn_relu_channels_first((merge9),32, 3)
    conv9 = conv_bn_relu_channels_first((conv9),32, 3)
    
    conv10 = Conv2D(1, 1, activation = 'sigmoid',data_format = 'channels_first')(conv9)
    addInToOut = Add(name = 'addInToOut')([inputs, conv10])     

    addInToOutTV = Add(name = 'addInToOutTV')([inputs, conv10])     

    model = Model(input = inputs, output = [addInToOut,addInToOutTV])    
    return model
    
#==========================================================================
#for evaluation
#==========================================================================
from PIL import Image

#This function is used to test the performance of 
import time
def pred_whole_folder_2_output_TV_NCHW_w_Xiaoke_performance(model, input_folder_name):
    #os.makedirs(output_folder_name,exist_ok=True)     
    start = time.time()
    for filename in os.listdir(input_folder_name): #assuming gif
        img = load_img(input_folder_name+filename,color_mode = "grayscale")
        img_size = img.size
        img = load_img(input_folder_name+filename,grayscale = True, target_size = (IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img)
        img = img/255    
        img = img.reshape((1,1,IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test = model.predict(img, batch_size=1, verbose=0)        
        #print(imgs_mask_test.shape)
        imgs_mask_test_side = imgs_mask_test[0].reshape((1, IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test_side = array_to_img(255 * imgs_mask_test_side, scale = False)#.resize(img_size,Image.ANTIALIAS)    
        #imgs_mask_test_side.save(output_folder_name + '/' + filename)   
        
        imgs_mask_test_final = imgs_mask_test[1].reshape((1, IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test_final = array_to_img(255 * imgs_mask_test_final, scale = False)#.resize(img_size,Image.ANTIALIAS)    
        #imgs_mask_test_final.save(output_folder_name + '/' + filename[:-4] + '_TV.png')   
    end = time.time()
    total_time = end - start
    print("\n"+ str(total_time))
        
def pred_whole_folder_2_output_TV_NCHW_w_Xiaoke(model, input_folder_name, output_folder_name):
    os.makedirs(output_folder_name,exist_ok=True)     
    for filename in os.listdir(input_folder_name): #assuming gif
        img = load_img(input_folder_name+filename,color_mode = "grayscale")
        img_size = img.size
        img = load_img(input_folder_name+filename,grayscale = True, target_size = (IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img)
        img = img/255    
        img = img.reshape((1,1,IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test = model.predict(img, batch_size=1, verbose=0)        
        
        #print(imgs_mask_test.shape)
        imgs_mask_test_side = imgs_mask_test[0].reshape((1, IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test_side = array_to_img(255 * imgs_mask_test_side, scale = False)#.resize(img_size,Image.ANTIALIAS)    
        imgs_mask_test_side.save(output_folder_name + '/' + filename)   
        
        #imgs_mask_test_final = imgs_mask_test[1].reshape((1, IMG_HEIGHT,IMG_WIDTH))
        #imgs_mask_test_final = array_to_img(255 * imgs_mask_test_final, scale = False)#.resize(img_size,Image.ANTIALIAS)    
        #imgs_mask_test_final.save(output_folder_name + '/' + filename[:-4] + '_TV.png')   
        #255 * imgs_mask_test, scale = False

import nibabel as nib
#resize nii
def pred_whole_folder_2_output_TV_NCHW_w_Xiaoke_nii(model, input_folder_name, output_folder_name):
    os.makedirs(output_folder_name,exist_ok=True)     
    for filename in os.listdir(input_folder_name): #assuming gif
        if filename.endswith(".nii"):
            img_nii = nib.load(input_folder_name+filename)
            img = img_nii.get_data()
            img = np.asarray(img)
            img = img/255    
            img = img.reshape((1,1,IMG_HEIGHT,IMG_WIDTH))
            imgs_mask_test = model.predict(img, batch_size=1, verbose=0)        
            imgs_mask_test_side = imgs_mask_test[0].reshape((1, IMG_HEIGHT,IMG_WIDTH))
            #imgs_mask_test_side = array_to_img(255 * imgs_mask_test_side, scale = False)#.resize(img_size,Image.ANTIALIAS)    
            imgs_mask_test_side  = np.squeeze(imgs_mask_test_side)
            imgs_mask_test_side = imgs_mask_test_side[:,:,np.newaxis]
            imgs_nii = nib.Nifti1Image(imgs_mask_test_side, img_nii.affine, img_nii.header)            
            save_path = output_folder_name+ '/' + filename
            nib.save(imgs_nii,save_path)
            #imgs_mask_test_side.save(output_folder_name + '/' + filename)   
            
def my_generator_NCHW(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255,        
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        #print(y_batch.shape)
        y_batch = np.transpose(y_batch,(0,3,1,2))
        x_batch = np.transpose(x_batch,(0,3,1,2))
        #print(y_batch.shape)
        yield x_batch, y_batch


def my_generator_2_output_NCHW(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255,        
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        y_batch = np.transpose(y_batch,(0,3,1,2))
        x_batch = np.transpose(x_batch,(0,3,1,2))
        
        yield x_batch, [y_batch, y_batch]
        
def my_generator_just_list_2_output_NCHW(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        y_batch = np.transpose(y_batch,(0,3,1,2))
        x_batch = np.transpose(x_batch,(0,3,1,2))
        
        yield x_batch, [y_batch, y_batch]
        
        
def my_generator_just_list_NCHW(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        #print(y_batch.shape)
        y_batch = np.transpose(y_batch,(0,3,1,2))
        x_batch = np.transpose(x_batch,(0,3,1,2))
        #print(y_batch.shape)
        yield x_batch, y_batch
        
def my_generator_2_output(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255,        
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        #print("generating images ")
        yield x_batch, [y_batch, y_batch]
        
        
        
        
def my_generator_just_list(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(MASK_LIB, class_mmy_generator_just_list_2_outputode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        yield x_batch, y_batch
        
        
def my_generator_just_list_2_output(IMAGE_LIB, MASK_LIB, _batch_size):
    data_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    mask_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED)
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        yield x_batch, [y_batch, y_batch]
        
def my_generator_save_images(IMAGE_LIB, MASK_LIB, _batch_size, IMAGE_DEST_LIB, MASK_DEST_LIB):
    data_generator = ImageDataGenerator(rescale=1./255,        
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(IMAGE_LIB,class_mode=None, batch_size=_batch_size, 
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED,save_to_dir=IMAGE_DEST_LIB, save_prefix='data_aug_image_', save_format='png')
    mask_generator = ImageDataGenerator(rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            zoom_range=0.1).flow_from_directory(MASK_LIB, class_mode=None, batch_size=_batch_size,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
                                                seed=SEED,save_to_dir=MASK_DEST_LIB, save_prefix='data_aug_label_', save_format='png')
    while True:
        x_batch= data_generator.next()
        y_batch= mask_generator.next()
        yield x_batch, y_batch
        

            
        '''
        img_size = img.size
        img = load_img(input_folder_name+filename,grayscale = True, target_size = (IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img)
        img = img/255    
        img = img.reshape((1,1,IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test = model.predict(img, batch_size=1, verbose=0)        
        
        #print(imgs_mask_test.shape)
        imgs_mask_test_side = imgs_mask_test[0].reshape((1, IMG_HEIGHT,IMG_WIDTH))
        imgs_mask_test_side = array_to_img(255 * imgs_mask_test_side, scale = False)#.resize(img_size,Image.ANTIALIAS)    
        imgs_mask_test_side.save(output_folder_name + '/' + filename)   
        '''
