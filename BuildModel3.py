# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 04:43:59 2018

@author: PC
"""
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.models import Sequential, Model, load_model
from keras.layers import (Permute, BatchNormalization, Dense, Dropout, 
AlphaDropout, Activation, LSTM, Bidirectional, Input, Conv2D, MaxPooling2D, 
Flatten, Embedding, Conv1D, MaxPooling1D, AveragePooling1D, Reshape,  
GlobalAveragePooling1D, GlobalAveragePooling2D, RNN, StackedRNNCells, LSTMCell, SeparableConv2D)
from keras.optimizers import SGD, RMSprop, adam
from keras.regularizers import L1L2
#import xception as xc


def build_model(index, indicator_length, data_row_count, interval_length, n_classes):

    if index == -1:
        
        dropout_val1 = 0.25
        dropout_val2 = 0.5
        reg = L1L2(l1=0.00001, l2=0.00001)
        
        data_input = Input(shape=(indicator_length, data_row_count, interval_length))
    
        c1 = Conv2D(128, (3, 3), kernel_regularizer=reg, use_bias=False)(data_input)
        c1 = AlphaDropout(dropout_val1)(c1)
        c1 = BatchNormalization()(c1)
        c1 = PReLU()(c1)
        
        c2 = Conv2D(128, (3, 3), kernel_regularizer=reg, use_bias=False)(c1)
        c2 = AlphaDropout(dropout_val1)(c2)
        c2 = BatchNormalization()(c2)
        c2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(c2)   
        c2 = GlobalAveragePooling2D()(c2)

#        residual_lstm = Permute((-1, 88, 128))(data_input)
#        residual_lstm = Reshape((-1, 88 * 128))(x) # 7, 88, 128
        lstm1 = Reshape((-1, indicator_length * data_row_count * interval_length))(data_input) # 7, 88, 128
        lstm1 = LSTM(128, return_sequences=True, 
                           dropout=dropout_val1, recurrent_dropout=dropout_val1,
                           implementation = 2, kernel_regularizer=reg)(lstm1)
        lstm1 = BatchNormalization()(lstm1)
        lstm2 = LSTM(128, return_sequences=True, 
                           dropout=dropout_val1, recurrent_dropout=dropout_val1,
                           implementation = 2, kernel_regularizer=reg)(lstm1)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = GlobalAveragePooling1D()(lstm2) 
        
        c1_lstm2 = Reshape((-1, 88 * 128))(c1)
        c1_lstm2 = LSTM(128, return_sequences=True, 
                           dropout=dropout_val1, recurrent_dropout=dropout_val1,
                           implementation = 2, kernel_regularizer=reg)(c1_lstm2)
        c1_lstm2 = BatchNormalization()(c1_lstm2)
        c1_lstm2 = GlobalAveragePooling1D()(c1_lstm2) 
        
        sc1 = SeparableConv2D(128, (3,3), padding='same', pointwise_regularizer=reg)(data_input)
        sc1 = AlphaDropout(dropout_val1)(sc1)
        sc1 = BatchNormalization()(sc1)
        sc1 = PReLU()(sc1)
        sc2 = SeparableConv2D(128, (3,3), padding='same', pointwise_regularizer=reg)(sc1)
        sc2 = AlphaDropout(dropout_val1)(sc2)
        sc2 = BatchNormalization()(sc2)
        sc2 = PReLU()(sc2)
        sc2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(sc2)
        sc2 = GlobalAveragePooling2D()(sc2)

        c1_sc2 = SeparableConv2D(128, (3,3), padding='same', pointwise_regularizer=reg)(c1)
        c1_sc2 = AlphaDropout(dropout_val1)(c1_sc2)
        c1_sc2 = BatchNormalization()(c1_sc2)
        c1_sc2 = PReLU()(c1_sc2)
        c1_sc2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(c1_sc2)
        c1_sc2 = GlobalAveragePooling2D()(c1_sc2)
        
        #sc1_lstm2 = Permute((-1, 88, 128))(sc1)
        sc1_lstm2 = Reshape((-1, 90 * 128))(sc1)
        sc1_lstm2 = LSTM(128, return_sequences=True, 
                           dropout=dropout_val1, recurrent_dropout=dropout_val1,
                           implementation = 2, kernel_regularizer=reg)(sc1_lstm2)
        sc1_lstm2 = BatchNormalization()(sc1_lstm2)
        sc1_lstm2 = GlobalAveragePooling1D()(sc1_lstm2) 
        
        #x = layers.concatenate([x, residual])
        a = layers.average([sc2, c2, lstm2])
        b = layers.maximum([sc2, c2, lstm2])
        c = layers.minimum([sc2, c2, lstm2])
        d = layers.add([sc2, c2, lstm2])
        e = layers.multiply([sc2, c2, lstm2])
        
        f = layers.average([c1_lstm2, c1_sc2, sc1_lstm2])
        g = layers.maximum([c1_lstm2, c1_sc2, sc1_lstm2])
        h = layers.minimum([c1_lstm2, c1_sc2, sc1_lstm2])
        i = layers.add([c1_lstm2, c1_sc2, sc1_lstm2])
        j = layers.multiply([c1_lstm2, c1_sc2, sc1_lstm2])
        
        x1 = layers.average([a,b,c,d,e])
        x2 = layers.average([f,g,h,i,j])
        
        x = layers.concatenate([x1,x2])
        
        #x = layers.concatenate([x, residual_lstm])
        #x = layers.average([x, residual_lstm])
        
        x = Dense(units=128, kernel_initializer='uniform')(x)
        x = AlphaDropout(dropout_val2)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Dense(units=128, kernel_initializer='uniform')(x)
        x = AlphaDropout(dropout_val2)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Dense(units=n_classes, activation='softmax')(x)

        model = Model(data_input, x)
        
        model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
        
        return model


    elif index == 0:
        
        dropout_val1 = 0.25
        dropout_val2 = 0.5
        reg = L1L2(l1=0.00001, l2=0.00001)
        
        data_input = Input(shape=(indicator_length, data_row_count, interval_length))
    
        x = Conv2D(128, (3, 3), kernel_regularizer=reg, use_bias=False)(data_input)
        x = AlphaDropout(dropout_val1)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        

#        residual_lstm = Permute((-1, 88, 128))(data_input)
#        residual_lstm = Reshape((-1, 88 * 128))(x) # 7, 88, 128
        residual_lstm = Reshape((-1, indicator_length * data_row_count * interval_length))(data_input) # 7, 88, 128
        residual_lstm = LSTM(128, return_sequences=True, 
                           dropout=dropout_val1, recurrent_dropout=dropout_val1,
                           implementation = 2, kernel_regularizer=reg)(residual_lstm)
        residual_lstm = BatchNormalization()(residual_lstm)
        residual_lstm = LSTM(128, return_sequences=True, 
                           dropout=dropout_val1, recurrent_dropout=dropout_val1,
                           implementation = 2, kernel_regularizer=reg)(residual_lstm)
        residual_lstm = BatchNormalization()(residual_lstm)
        residual_lstm = GlobalAveragePooling1D()(residual_lstm)



        residual = Conv2D(128, (1, 1), strides=(1, 1),
                      padding='same', kernel_regularizer=reg, use_bias=False)(x)
        residual = AlphaDropout(dropout_val1)(residual)
        residual = BatchNormalization()(residual)       

        
        x = SeparableConv2D(128, (3,3), padding='same', pointwise_regularizer=reg)(x)
        x = AlphaDropout(dropout_val1)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        
        x = SeparableConv2D(128, (3,3), padding='same', pointwise_regularizer=reg)(x)
        x = AlphaDropout(dropout_val1)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        
        x = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        
        x = layers.average([x, residual])
        #x = layers.concatenate([x, residual])
        

        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)

        #x = layers.concatenate([x, residual_lstm])
        x = layers.average([x, residual_lstm])
        
        x = Dense(units=128, kernel_initializer='uniform')(x)
        x = AlphaDropout(dropout_val2)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Dense(units=128, kernel_initializer='uniform')(x)
        x = AlphaDropout(dropout_val2)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Dense(units=n_classes, activation='softmax')(x)

        model = Model(data_input, x)
        
        model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
        
        return model


#    elif index == 1:
        
#        model = xc.Xception(include_top=True, weights=None,
#                            input_shape=(indicator_length, data_row_count, interval_length),
#                            pooling='avg', classes=n_classes, noisy=True, dropout_val=0.4, 
#                            reg_val=0.00005)
           
#        model.compile(loss='categorical_crossentropy',
#                      optimizer='nadam',
#                      metrics=['accuracy'])
        
#        model.compile(loss='categorical_crossentropy',
#                      optimizer=adam(amsgrad=True),
#                      metrics=['accuracy'])
        
#        return model

#    elif index == 2:
        
#        model = xc.Xception(include_top=True, weights=None,
#                            input_shape=(indicator_length, data_row_count, interval_length),
#                            pooling='avg', classes=n_classes, noisy=True, dropout_val=0.25,
#                            reg_val=0.00001, prlu=False, stride=1)
#           
#        model.compile(loss='categorical_crossentropy',
#                      optimizer='nadam',
#                      metrics=['accuracy'])
#        
#        model.compile(loss='categorical_crossentropy',
#                      optimizer=adam(amsgrad=True),
#                      metrics=['accuracy'])
#        
#        return model

