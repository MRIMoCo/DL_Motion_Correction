#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from data import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
#from keras.losses import mean_square_error
from keras.models import Model
from util_unet_NCHW_clean import *
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import nibabel as nib
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from keras.optimizers import Adam
model = UNet_Total_Variation_BN_NCHW(IMG_HEIGHT, IMG_WIDTH)
model.summary()
#The location of the model
save_dir = './Models/UNet_exp_20_s8_L1_plus_TV_batch_size_8_ft/'
def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    pre_fn = []
    if file_list:
        epochs_exist = []
        file_list = sorted(file_list)
        for file_ in (file_list):
#            print(file_)
            result = re.findall(".*model_(.*).hdf5.*",file_)
            #print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)
        arg_idx = np.argmax(epochs_exist)
        print('model file name', file_list[arg_idx])
        pre_fn = file_list[arg_idx]
    else:
        initial_epoch = 0
    return initial_epoch, pre_fn
pretrained_dir =save_dir
_, trained_model_fn = findLastCheckpoint(pretrained_dir)

model.load_weights(trained_model_fn)
#input folder 
input_dir =   './Images/corrupted/'
#Results folder
result_dir = './Results_folder/' 
#pred_whole_folder_2_output_TV_NCHW_w_Xiaoke(model, input_folder_corrupt, output_folder_corrupt)
pred_whole_folder_2_output_TV_NCHW_w_Xiaoke_nii(model, input_dir, result_dir)