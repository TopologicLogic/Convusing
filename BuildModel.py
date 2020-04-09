# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 04:43:59 2018

@author: PC
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential, Model, load_model
from keras.layers import (Permute, BatchNormalization, Dense, Dropout, 
AlphaDropout, Activation, LSTM, Bidirectional, Input, Conv2D, MaxPooling2D, 
Flatten, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, 
GlobalAveragePooling1D, GaussianNoise, LocallyConnected1D)
from keras.optimizers import SGD, RMSprop, adam
from keras.regularizers import L1L2


def build_model(index, data_column_count, data_row_count, n_classes):
    
    if index == 0:
        rm1 = 2
        rm2 = 2
        dropout_val1 = 0.25
        dropout_val2 = 0.5

        model = Sequential()

        model.add(BatchNormalization(input_shape=(data_column_count, data_row_count)))
        
        model.add(Conv1D(256, 9, padding='same', 
                         input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(Conv1D(256, 9, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(Bidirectional(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       input_shape=(data_column_count, data_row_count))))
        model.add(Dropout(dropout_val1))
        model.add(Bidirectional(LSTM(round(data_row_count * rm1), return_sequences=True)))
        model.add(Dropout(dropout_val1))
    
        model.add(Flatten())
    #    model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
    #    model.add(PReLU())
    #    model.add(Dropout(dropout_val2))
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=n_classes, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
    
        return model
    
    elif index == 1:
        rm1 = 2
        rm2 = 2
        dropout_val1 = 0.25
        dropout_val2 = 0.5

        model = Sequential()

        model.add(Conv1D(256, 3, padding='same', 
                  input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(256, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))

        model.add(BatchNormalization())
        
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True))
        model.add(Dropout(dropout_val1))
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True))
        model.add(Dropout(dropout_val1))
    
        model.add(BatchNormalization())
    
        model.add(Flatten())
        
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=n_classes, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
    
        return model
    
    elif index == 2:
        rm1 = 2
        rm2 = 2
        dropout_val1 = 0.25
        dropout_val2 = 0.5
        
        model = Sequential()

        model.add(Conv1D(256, 3, padding='same', 
                  input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(256, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(256, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))

        model.add(BatchNormalization())
        
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2))
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2))
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2))
    
        model.add(BatchNormalization())
    
        model.add(Flatten())
        
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=n_classes, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
    
        return model
    
    elif index == 3:
        rm1 = 3
        rm2 = 3
        dropout_val1 = 0.5
        dropout_val2 = 0.5
        reg = L1L2(l1=0.0007, l2=0.0007)
        
        model = Sequential()

        model.add(Conv1D(5040, 3, padding='same', 
                  input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))

        model.add(BatchNormalization())
        
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2, kernel_regularizer=reg))
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2, kernel_regularizer=reg))
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2, kernel_regularizer=reg))
        model.add(LSTM(round(data_row_count * rm1), return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 2, kernel_regularizer=reg))
        
        model.add(BatchNormalization())
    
        model.add(Flatten())
        
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=round(data_row_count * rm2), kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=n_classes, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
    
        return model

    elif index == 4:

        dropout_val1 = 0.45
        dropout_val2 = 0.5
        reg = L1L2(l1=0.0007, l2=0.0007)
        
        model = Sequential()

        model.add(Conv1D(5040, 3, padding='same', 
                  input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))
        
        model.add(Conv1D(256, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(256, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(256, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(BatchNormalization())
        
        model.add(LSTM(512, return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 1, kernel_regularizer=reg))
        model.add(LSTM(512, return_sequences=True, 
                       dropout=dropout_val1, recurrent_dropout=dropout_val1,
                       implementation = 1, kernel_regularizer=reg))
        
        model.add(BatchNormalization())
    
        model.add(Flatten())
        
        model.add(Dense(units=512, kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=256, kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=256, kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=n_classes, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy',
                      optimizer='nadam',
                      metrics=['accuracy'])
    
        return model

    elif index == 5:

        dropout_val1 = 0.40
        dropout_val2 = 0.5
        reg = L1L2(l1=0.0005, l2=0.0005)
        
        model = Sequential()
        
        model.add(Conv1D(512, 3, padding='same', 
              input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
            
        model.add(BatchNormalization())
        
        model.add(LSTM(384, return_sequences=True, 
                   dropout=dropout_val1, recurrent_dropout=dropout_val1,
                   implementation = 2, kernel_regularizer=reg))
        model.add(LSTM(384, return_sequences=True, 
                   dropout=dropout_val1, recurrent_dropout=dropout_val1,
                   implementation = 2, kernel_regularizer=reg))
        model.add(LSTM(384, return_sequences=True, 
                   dropout=dropout_val1, recurrent_dropout=dropout_val1,
                   implementation = 2, kernel_regularizer=reg))
        
        model.add(BatchNormalization())
        
        model.add(Flatten())
        
        model.add(Dense(units=256, kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=256, kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=256, kernel_initializer='uniform'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val2))
        model.add(Dense(units=n_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
        
        return model
        
    elif index == 6:

        dropout_val1 = 0.5
        dropout_val2 = 0.5
        
        model = Sequential()
        
        model.add(Conv1D(512, 3, padding='same', 
              input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))        
        model.add(BatchNormalization())
        
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding='same'))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(BatchNormalization())
        
        model.add(Flatten())

        model.add(Dense(units=n_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
        
        return model        

    elif index == 7:

        pad = 'causal'
        dr = 2
        reg = L1L2(l1=0.0005, l2=0.0005)
        dropout_val1 = 0.5
        dropout_val2 = 0.5
        init = 'glorot_uniform'
        
        model = Sequential()
        
        model.add(GaussianNoise(0.5, input_shape=(data_column_count, data_row_count)))
        
#        model.add(LocallyConnected1D(512, 3, kernel_regularizer=reg, 
#              kernel_initializer=init))
#        model.add(PReLU())
#        model.add(AlphaDropout(dropout_val1))
#        model.add(LocallyConnected1D(512, 3, kernel_regularizer=reg, 
#                         kernel_initializer=init))
#        model.add(PReLU())
#        model.add(AlphaDropout(dropout_val1))
#        model.add(LocallyConnected1D(512, 3, kernel_regularizer=reg, 
#                         kernel_initializer=init))
#        model.add(PReLU())
#        model.add(AlphaDropout(dropout_val1))
#        model.add(LocallyConnected1D(512, 3, kernel_regularizer=reg, 
#                         kernel_initializer=init))
#        model.add(PReLU())
#        model.add(AlphaDropout(dropout_val1))
        
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init, input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init, input_shape=(data_column_count, data_row_count)))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))
        model.add(BatchNormalization())
        
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(MaxPooling1D(2))        
        model.add(BatchNormalization())
        
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        model.add(Conv1D(512, 3, padding=pad, dilation_rate=dr, kernel_regularizer=reg, 
                         kernel_initializer=init))
        model.add(PReLU())
        model.add(AlphaDropout(dropout_val1))
        
        model.add(GlobalAveragePooling1D())
        
        model.add(BatchNormalization())
        
#        model.add(Flatten())
        
        

        model.add(Dense(units=n_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
        
        return model   








