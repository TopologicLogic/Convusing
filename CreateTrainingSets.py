# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 09:06:28 2018

@author: PC
"""
import os
import DataPrep as dp
import numpy as np

n_classes = 21
batch_size = 64
batch_mult = 20
data_row_count = 180
data_column_count = 28
projection_range = 0 # 0 = One day look-ahead, 4 = Five day look-ahead

if not os.path.exists('TrainingData/'): os.makedirs('TrainingData/')

for i in range(13, 100):
    print("\nCreating batch #" + str(i+1) + "...\n")
    x_train, y_train = dp.homogeneous_populate_training(n_classes, batch_size * batch_mult, 
                                                data_column_count, data_row_count, 
                                                projection_range, check_for_zeros=False)
    np.save("TrainingData/x_train-" + str(i), x_train, allow_pickle=False)
    np.save("TrainingData/y_train-" + str(i), y_train, allow_pickle=False)

for i in range(100, 200):
    print("\nCreating batch #" + str(i+1) + "...\n")
    x_train, y_train = dp.random_normal_populate_training(n_classes, batch_size * batch_mult, 
                                                data_column_count, data_row_count, 
                                                projection_range, (n_classes-1) / 2, 3, 
                                                check_for_zeros=False)
    np.save("TrainingData/x_train-" + str(i), x_train, allow_pickle=False)
    np.save("TrainingData/y_train-" + str(i), y_train, allow_pickle=False)