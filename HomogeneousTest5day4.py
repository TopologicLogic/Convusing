# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:47:40 2018

@author: PC
"""
import sys
import os.path
import random
import BuildModel3 as bm3

import numpy as np
from keras.models import load_model
from keras.utils import plot_model

n_classes = 21
batch_size = 64
batch_mult = 20
data_row_count = 90
projection_range = 0 # 0 = One day look-ahead, 4 = Five day look-ahead
n_batches = 200
model_index = -1
model_name = 'stocks5-5day(index=' + str(model_index) + ').h5'

while True:

    model = object()
    batches = {}
    batches_test = {}
    
    indicators = ['rsi', 'atr', 'wr', 'vr', 'beta', 'alpha', \
                  'rsquared', 'volatility', 'momentum']
    intervals = ['2', '5', '10', '15', '20', '30', '60']
               
    
    # Reset the model to it's last 'good' state, or create a new one.
    if os.path.isfile(model_name):
        print("Loading model...\n")
        model = load_model(model_name)
        plot_model(model, to_file=model_name + '.png')
    else:
        print("Building model...\n")
        model = bm3.build_model(model_index, len(indicators), data_row_count, len(intervals), 
                                n_classes)

        
#    batch_div = np.floor(n_batches / 2)
#
#    for i in range(0, n_batches):
#        if i < batch_div:
#            batch_i = np.random.randint(0, 100)
#            print("Loading batch #" + str(batch_i+1))
#            x_t = np.load('TrainingData5day3/x_train-' + str(batch_i) + '.npy')
#            y_t = np.load('TrainingData5day3/y_train-' + str(batch_i) + '.npy')
#            batches[i] = x_t, y_t
#        else:
#            batch_i = np.random.randint(100, 200)
#            print("Loading batch #" + str(batch_i+1))
#            x_t = np.load('TrainingData5day3/x_train-' + str(batch_i) + '.npy')
#            y_t = np.load('TrainingData5day3/y_train-' + str(batch_i) + '.npy')
#            batches[i] = x_t, y_t
    
    for i in range(0, n_batches):
        print("Loading batch #" + str(i+1) + " of " + str(n_batches))
        x_t = np.load('TrainingData5day3/x_train-' + str(i) + '.npy')
        y_t = np.load('TrainingData5day3/y_train-' + str(i) + '.npy')
        batches[i] = x_t, y_t
        
    random.shuffle(batches)
    
    big_x = np.zeros((batch_size * batch_mult * len(batches), len(indicators), data_row_count, len(intervals)))
    big_y = np.zeros((batch_size * batch_mult * len(batches), n_classes))
    
    z = 0
    for i in range(0, len(batches)):
        tx, ty = batches[i]
        for q in range(0, batch_size * batch_mult):
            big_x[z] = tx[q]
            big_y[z] = ty[q]
            z += 1
        
    start1 = np.random.randint(0, 75)
    start2 = np.random.randint(100, 175)
    for i in range(0, 25):
        print("Loading test batch #" + str(i+1+start1))
        x_t = np.load('TrainingData5day3-NYSE/x_train-' + str(i+start1) + '.npy')
        y_t = np.load('TrainingData5day3-NYSE/y_train-' + str(i+start1) + '.npy')
        batches_test[i] = x_t, y_t
    for i in range(0, 25):
        print("Loading test batch #" + str(i+1+start2))
        x_t = np.load('TrainingData5day3-NYSE/x_train-' + str(i+start2) + '.npy')
        y_t = np.load('TrainingData5day3-NYSE/y_train-' + str(i+start2) + '.npy')
        batches_test[i+25] = x_t, y_t
                  
    random.shuffle(batches_test)
    
    big_x_test = np.zeros((batch_size * batch_mult * len(batches_test), len(indicators), data_row_count, len(intervals)))
    big_y_test = np.zeros((batch_size * batch_mult * len(batches_test), n_classes))
    
    z = 0
    for i in range(0, len(batches_test)):
        tx, ty = batches_test[i]
        for q in range(0, batch_size * batch_mult):
            big_x_test[z] = tx[q]
            big_y_test[z] = ty[q]
            z += 1
    
        
    print("Training started...\n")
    
    dup_stop = 20 # Number of duplicate losses to stop after.
    dup_stop_threshold = 0.00001
    losshist = []
    best = sys.float_info.min
    
    for q in range(0, 5000):

        print("Epoch: " + str(q+1) + ", model_index=" + str(model_index))
        status = model.fit(big_x, big_y, epochs=1, batch_size=128,
                           shuffle=True, validation_data=(big_x_test,big_y_test),
                           verbose=1)
        
        val_loss = status.history['val_loss'][0]
        val_acc = status.history['val_acc'][0]
        loss = status.history['loss'][0]
        acc = status.history['acc'][0]
        
        #if val_acc >= 0.85: 
        if val_acc > best:
            print("Saving model...\n")
            model.save(model_name, overwrite=True)
            with open(model_name + ".txt", "w") as text_file:
                text_file.write("epoch\tloss\tacc\tval_loss\tval_acc\n" + 
                                str(q) + "\t" + str(loss) + "\t" + str(acc) + "\t" + 
                                str(val_loss) + "\t" + str(val_acc) + "\n") 
            best = val_acc
            if val_acc >= 0.99: break
        
        losshist.append(loss)
        
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
    