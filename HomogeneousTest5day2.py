# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:47:40 2018

@author: PC
"""
import sys
import os.path
import random
import BuildModel as bm
import DataPrep as dp

import numpy as np
from keras.models import load_model
from keras.utils import plot_model

n_classes = 21
batch_size = 64
batch_mult = 20
data_row_count = 90
data_column_count = 28 + 35
projection_range = 4 # 0 = One day look-ahead, 4 = Five day look-ahead
n_batches = 200
model_index = 6
model_name = 'stocks4-5day(index=' + str(model_index) + ').h5'

while True:

    batches = {}
    
    # Reset the model to it's last 'good' state, or create a new one.
    if os.path.isfile(model_name):
        print("Loading model...\n")
        model = load_model(model_name)
        #plot_model(model, to_file=model_name + '.png')
    else:
        print("Building model...\n")
        model = bm.build_model(model_index, data_column_count, data_row_count, n_classes)
        
    for i in range(0, n_batches):
        print("Loading batch #" + str(i+1) + " of " + str(n_batches))
        x_t = np.load('TrainingData5day2/x_train-' + str(i) + '.npy')
        y_t = np.load('TrainingData5day2/y_train-' + str(i) + '.npy')
        batches[i] = x_t, y_t

#    batch_div = np.floor(n_batches / 2)
#
#    for i in range(0, n_batches):
#        if i < batch_div:
#            batch_i = np.random.randint(0, 100)
#            print("Loading batch #" + str(batch_i+1))
#            x_t = np.load('TrainingData5day2/x_train-' + str(batch_i) + '.npy')
#            y_t = np.load('TrainingData5day2/y_train-' + str(batch_i) + '.npy')
#            batches[i] = x_t, y_t
#        else:
#            batch_i = np.random.randint(100, 200)
#            print("Loading batch #" + str(batch_i+1))
#            x_t = np.load('TrainingData5day2/x_train-' + str(batch_i) + '.npy')
#            y_t = np.load('TrainingData5day2/y_train-' + str(batch_i) + '.npy')
#            batches[i] = x_t, y_t
    
    random.shuffle(batches)
    
    big_x = np.zeros((batch_size * batch_mult * len(batches), data_column_count, data_row_count))
    big_y = np.zeros((batch_size * batch_mult * len(batches), n_classes))
    
    z = 0
    for i in range(0, len(batches)):
        tx, ty = batches[i]
        for q in range(0, batch_size * batch_mult):
            big_x[z] = tx[q]
            big_y[z] = ty[q]
            z += 1
    
   
#    batch_i = np.random.randint(0, n_batches)
#    print("Loading batch #" + str(batch_i+1) + " of " + str(n_batches))
#    x_train = np.load('TrainingData5day2/x_train-' + str(batch_i) + '.npy')
#    y_train = np.load('TrainingData5day2/y_train-' + str(batch_i) + '.npy')
    
    print("Training started...\n")
    
    dup_stop = 20 # Number of duplicate losses to stop after.
    dup_stop_threshold = 0.00001
    losshist = []
    best = sys.float_info.min
    
    for q in range(0, 500):

#        batch_i = np.random.randint(0, len(batches))
#        x_train, y_train = batches[batch_i]
#        print("Batch #" + str(batch_i+1) + " of " + str(n_batches) + ", Epoch: " + str(q+1))
#        status = model.fit(x_train, y_train, epochs=1, batch_size=batch_size * batch_mult)
#        
        print("Epoch: " + str(q+1) + ", model_index=" + str(model_index))
        status = model.fit(big_x, big_y, epochs=1, batch_size=batch_size,
                           verbose=2)

        loss = status.history['loss'][0]
        acc = status.history['acc'][0]
        
#        model.save("stocks4-5day(index=" + str(model_index) + "; epoch=" + str(q+1) + 
#                                 "; acc=" + str(acc) + ").h5", overwrite=True)
        
        losshist.append(loss)  
        
        if acc >= 0.85: 
            if acc > best:
                print("Saving model...\n")
                model.save(model_name, overwrite=True)
                best = acc
            if acc >= 0.98: break
        
        if len(losshist) > dup_stop + 1:
            alldups = True
            for z in range(len(losshist) - 2, len(losshist) - dup_stop - 1, -1):
                if abs(loss - losshist[z]) > dup_stop_threshold:
                    alldups = False
                    break
            if alldups: 
                print("Too many duplicate outcomes in a row, stopping...\n")
                break
    

    del model
    del x_train
    del y_train
    del batches
    