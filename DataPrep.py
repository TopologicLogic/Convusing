# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:19:35 2018

@author: Dev
"""
import random
import pandas as pan
import numpy as np
import os.path
import csv
import sys
#import ctypes  # An included library with Python install.   

from copy import deepcopy
from stockstats import StockDataFrame as sdf
from googlefinance.client import get_price_data #, get_prices_data, get_prices_time_data

data_lookup = {}

def find_intervals(projection_range, row_start, sdfdata, interval_min, interval_max):
    interval_index = []
    
    #tshape = sdfdata.iloc[:,0:].values.shape
    #if row_start >= tshape[0]: return interval_index
        
    tdata = sdfdata['close_-' + str(projection_range+1) +'_r'].as_matrix()
#    print(str(tdata))
    for i in range(row_start, len(tdata)):
        if tdata[i] >= interval_min and tdata[i] <= interval_max:
            interval_index.append(i)
        
    return interval_index



def get_indicator_data_at_index(index, data_row_count, sdfdata, indicators, intervals, normalize=True):
    
    x_data = np.zeros((len(indicators) * len(intervals), data_row_count))
    l = 0    
    
    if normalize:
       
        for m in range(0, len(indicators)):
            for n in range(0, len(intervals)):
                # A one dimensional array with the indicator/interval data
                tdata = sdfdata[indicators[m] + '_' + intervals[n]].as_matrix()

                dmax = sys.float_info.min
                
                # Check all of tdata for NaNs, Infinities, and get column max
                for t in range(0, len(tdata)):
                    if not np.isfinite(tdata[t]): tdata[t] = 0
                    if tdata[t] > dmax: dmax = tdata[t]
                
                for t in range(0, len(tdata)):
                    tdata[t] /= dmax
                
                      
                q = 0
                for t in range(index - data_row_count, index):                   
                    x_data[l][q] = tdata[t]
                    q += 1
                    
                l += 1
                
                
    else:
        
        for m in range(0, len(indicators)):
            for n in range(0, len(intervals)):
                tdata = sdfdata[indicators[m] + '_' + intervals[n]].as_matrix()
                q = 0
                for t in range(index - data_row_count, index):
                    x_data[l][q] = tdata[t]
                    if not np.isfinite(x_data[l][q]): x_data[l][q] = 0
                    q += 1
                l += 1

    return x_data



def homogeneous_populate_training(n_classes, batch_count, data_column_count, data_row_count,
                                  projection_range, check_for_zeros=True, track_classes=True, 
                                  verbose=True):

    global data_lookup
    
    indicators = ['rsi', 'atr', 'wr', 'vr']
    intervals = ['2', '5', '10', '15', '20', '30', '60']
           
    x_train = np.zeros((batch_count, data_column_count, data_row_count))
    y_train = np.zeros((batch_count, n_classes), dtype=int)
    
    symbols = []

    
    #https://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download
    with open('NASD.csv', newline='') as csvfile:
        r = csv.DictReader(csvfile)
        for row in r:
            symbols.append(row['Symbol'])
           
            
    class_max = [-50,    -20, -15, -10, -5,  -4, -3, -2, -1, -0.1, 0,    1, 2, 3, 4, 5, 10, 15, 20, 50, 10000]
    class_min = [-10000, -50, -20, -15, -10, -5, -4, -3, -2, -1,   -0.1, 0, 1, 2, 3, 4, 5,  10, 15, 20, 50   ]
    
    avg_class_count = np.floor((batch_count) / n_classes)
    
    class_count = np.zeros(n_classes, dtype=int)
    class_count.fill(avg_class_count)
    
    diff = batch_count - (avg_class_count * n_classes)
    if diff > 0: class_count[10] += diff
    
    if not os.path.exists('StockData/'): os.makedirs('StockData/')
    
    batch_i = 0
    
    for z in range(0, n_classes):
        
        skips = [] # Skiped due to data errors
        
        nics = [] # Not in class
        
        if track_classes:
            if os.path.isfile("skips.csv"):
                with open('skips.csv', newline='\n') as csvfile:
                    r = csv.DictReader(csvfile)
                    for row in r: skips.append(row['Symbols'])            
            else:
                with open("skips.csv", "w") as text_file:
                    text_file.write("""Symbols""" + "\n") 
                    
            if os.path.isfile("nic-" + str(z) + ".csv"):
                with open("nic-" + str(z) + ".csv", newline='\n') as csvfile:
                    r = csv.DictReader(csvfile)
                    for row in r: nics.append(row['Symbols'])
            else:
                with open("nic-" + str(z) + ".csv", "w") as text_file:
                    text_file.write("""Symbols""" + "\n")                   

            if verbose:
                print("\nClass #" + str(z+1) + " of " + str(n_classes) + 
                      ", Total skips: " + str(len(skips) + len(nics)) + "\n")
        elif verbose:
            print("\nClass #" + str(z) + " of " + str(n_classes))


        symbols_used = []
        
        k = 0
        while k < class_count[z]:
            
            tclass = z
            tsymbol = symbols[np.random.randint(0, len(symbols))]
            
            # Went through all the symbols and cound't find enough examples,
            # so fill up more default values.
            if len(symbols_used) >= len(symbols):
                tclass = 10
            
            # Check for stocks to skip
            if track_classes: 
                if tsymbol in skips: continue
                if tsymbol in nics: continue
            
            
            if tsymbol in symbols_used: continue
            symbols_used.append(tsymbol)
                        
            tshape = object()

            if tsymbol in data_lookup:
                if verbose:
                    print("Loading["+ str(class_min[tclass]) + "%:" + str(class_max[tclass]) + "%, " +
                                   "#" + str(k) + "]: " + tsymbol)
                df = data_lookup[tsymbol]
            elif os.path.isfile("StockData/" + tsymbol + ".csv"):
                if verbose:
                    print("Loading["+ str(class_min[tclass]) + "%:" + str(class_max[tclass]) + "%, " +
                                   "#" + str(k) + "]: " + tsymbol)
                data_lookup[tsymbol] = pan.read_csv("StockData/" + tsymbol + ".csv", 
                                                    sep=',', header=0, index_col=0)
                df = data_lookup[tsymbol]
            else:
                if verbose: 
                    print("Downloading["+ str(class_min[tclass]) + "%:" + str(class_max[tclass]) + "%, " +
                                       "#" + str(k) + "]: " + tsymbol)
                param = {
                'q': tsymbol, # Stock symbol (ex: "AAPL")
                'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
                'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
                'p': "5Y" # Period (Ex: "1Y" = 1 year)
                }       
                # get price data (return pandas dataframe)
                df = get_price_data(param)                
                df.to_csv("StockData/" + tsymbol + ".csv")
                data_lookup[tsymbol] = deepcopy(df)
            
            tshape = df.iloc[:,0:].values.shape
        
            if tshape[0] <= data_row_count: 
                if verbose: 
                    print("Data error: " + tsymbol + 
                          ", continuing to next symbol...")
                if track_classes:
                    with open("skips.csv", "a") as text_file:
                            text_file.write(tsymbol + "\n")                   
                continue
        
            if tshape[0] < 400: 
                if verbose: 
                    print("Not enough data for: " + tsymbol +
                          ", continuing to next symbol...")
                if track_classes:
                    with open("skips.csv", "a") as text_file:
                            text_file.write(tsymbol + "\n")
                continue
            
            # Check for zeros
            if check_for_zeros:
                zero_flag = False
                for row in range(0, tshape[0]):
                    for column in range(0, tshape[1]):
                            v = df.iloc[:,column:].values[row][0]
                            if v <= 0:
                                zero_flag = True
                                break
                    if zero_flag: break
                if zero_flag: 
                    if verbose: 
                        print("Zeros in: " + tsymbol + 
                              ", continuing to next symbol...")
                    if track_classes:
                        with open("skips.csv", "a") as text_file:
                                text_file.write(tsymbol + "\n")               
                    continue
            
            
            sdfdata = sdf.retype(df)
                  
            indicies = find_intervals(projection_range, data_row_count + projection_range, 
                                      sdfdata, class_min[tclass], class_max[tclass])
        
            if len(indicies) > 0:
                random.shuffle(indicies)
                for i in range(0, len(indicies)):
                    if k < class_count[z] and batch_i < batch_count: 
                        # Add data to the batch array
                        x_train[batch_i] = get_indicator_data_at_index(indicies[i]-projection_range, 
                                           data_row_count, sdfdata, indicators, intervals)
                        y_train[batch_i][tclass] = 1
                        k += 1
                        batch_i += 1
                    else:
                        break
            elif track_classes:
                with open("nic-" + str(z) + ".csv", "a") as text_file:
                    text_file.write(tsymbol + "\n")                   
            
            del sdfdata
            del df
        
        del symbols_used
        del skips
        del nics

    return x_train, y_train



def homogeneous_populate_training2(n_classes, batch_count, data_column_count, data_row_count,
                                  projection_range, check_for_zeros=True, track_classes=True, 
                                  verbose=True):

    global data_lookup

    indicators = ['rsi', 'atr', 'wr', 'vr']
    intervals = ['2', '5', '10', '15', '20', '30', '60']
           
    x_train = np.zeros((batch_count, data_column_count, data_row_count))
    y_train = np.zeros((batch_count, n_classes), dtype=int)
    
    symbols = []

    
    #https://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download
    with open('NASD.csv', newline='') as csvfile:
        r = csv.DictReader(csvfile)
        for row in r:
            symbols.append(row['Symbol'])
            
    class_max = [-50,    -20, -15, -10, -5,  -4, -3, -2, -1, -0.1, 0,    1, 2, 3, 4, 5, 10, 15, 20, 50, 10000]
    class_min = [-10000, -50, -20, -15, -10, -5, -4, -3, -2, -1,   -0.1, 0, 1, 2, 3, 4, 5,  10, 15, 20, 50   ]
    
    avg_class_count = np.floor((batch_count) / n_classes)
    
    class_count = np.zeros(n_classes, dtype=int)
    class_count.fill(avg_class_count)
    
    # If the data is uneven, put it in the center.
    diff = batch_count - (avg_class_count * n_classes)
    if diff > 0: class_count[10] += diff
    
    if not os.path.exists('StockData/'): os.makedirs('StockData/')
    
    batch_i = 0
    
    for z in range(0, n_classes):
        
        skips = [] # Skiped due to data errors
        
        nics = [] # Not in class
        
        if track_classes:
            if os.path.isfile("skips.csv"):
                with open('skips.csv', newline='\n') as csvfile:
                    r = csv.DictReader(csvfile)
                    for row in r: skips.append(row['Symbols'])            
            else:
                with open("skips.csv", "w") as text_file:
                    text_file.write("""Symbols""" + "\n") 
                    
            if os.path.isfile("nic-" + str(z) + ".csv"):
                with open("nic-" + str(z) + ".csv", newline='\n') as csvfile:
                    r = csv.DictReader(csvfile)
                    for row in r: nics.append(row['Symbols'])
            else:
                with open("nic-" + str(z) + ".csv", "w") as text_file:
                    text_file.write("""Symbols""" + "\n")                   

            if verbose:
                print("\nClass #" + str(z+1) + " of " + str(n_classes) + 
                      ", Total skips: " + str(len(skips) + len(nics)) + "\n")
        elif verbose:
            print("\nClass #" + str(z) + " of " + str(n_classes))


        random.shuffle(symbols)

        for k in range(0, symbols):
            
            tclass = z
            tsymbol = symbols[k]
            
            # Check for stocks to skip
            if track_classes: 
                if tsymbol in skips: continue
                if tsymbol in nics: continue
          
            if os.path.isfile("StockData/" + tsymbol + ".csv"):
                data_lookup[tsymbol] = pan.read_csv("StockData/" + tsymbol + ".csv", 
                                                    sep=',', header=0, index_col=0)
            
            if tsymbol in data_lookup:
                if verbose:
                    print("Loading["+ str(class_min[tclass]) + "%:" + str(class_max[tclass]) + "%, " +
                                   "#" + str(k) + "]: " + tsymbol)
                df = data_lookup[tsymbol]
            else:
                if verbose: 
                    print("Downloading["+ str(class_min[tclass]) + "%:" + str(class_max[tclass]) + "%, " +
                                       "#" + str(k) + "]: " + tsymbol)
                param = {
                'q': tsymbol, # Stock symbol (ex: "AAPL")
                'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
                'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
                'p': "5Y" # Period (Ex: "1Y" = 1 year)
                }       
                # get price data (return pandas dataframe)
                df = get_price_data(param)                
                df.to_csv("StockData/" + tsymbol + ".csv")
                data_lookup[tsymbol] = deepcopy(df)
            
            tshape = df.iloc[:,0:].values.shape
        
            if tshape[0] <= data_row_count: 
                if verbose: 
                    print("Data error: " + tsymbol + 
                          ", continuing to next symbol...")
                if track_classes:
                    with open("skips.csv", "a") as text_file:
                            text_file.write(tsymbol + "\n")                   
                continue
        
            if tshape[0] < 400: 
                if verbose: 
                    print("Not enough data for: " + tsymbol +
                          ", continuing to next symbol...")
                if track_classes:
                    with open("skips.csv", "a") as text_file:
                            text_file.write(tsymbol + "\n")
                continue
            
            # Check for zeros
            if check_for_zeros:
                zero_flag = False
                for row in range(0, tshape[0]):
                    for column in range(0, tshape[1]):
                            v = df.iloc[:,column:].values[row][0]
                            if v <= 0:
                                zero_flag = True
                                break
                    if zero_flag: break
                if zero_flag: 
                    if verbose: 
                        print("Zeros in: " + tsymbol + 
                              ", continuing to next symbol...")
                    if track_classes:
                        with open("skips.csv", "a") as text_file:
                                text_file.write(tsymbol + "\n")               
                    continue
            
            sdfdata = sdf.retype(df)
                  
            indicies = find_intervals(projection_range, data_row_count + projection_range, 
                                      sdfdata, class_min[tclass], class_max[tclass])
        
            if len(indicies) > 0:
                random.shuffle(indicies)
                for i in range(0, len(indicies)):
                    if k < class_count[z] and batch_i < batch_count: 
                        # Add data to the batch array
                        x_train[batch_i] = get_indicator_data_at_index(indicies[i]-projection_range, 
                                           data_row_count, sdfdata, indicators, intervals)
                        y_train[batch_i][tclass] = 1
                        k += 1
                        batch_i += 1
                    else:
                        break
            elif track_classes:
                with open("nic-" + str(z) + ".csv", "a") as text_file:
                    text_file.write(tsymbol + "\n")                   
            
            del sdfdata
            del df
        
        del skips
        del nics

    return x_train, y_train



def random_normal_populate_training(n_classes, batch_count, data_column_count, data_row_count,
                            projection_range, mu, sigma, check_for_zeros=True, 
                            track_classes=True,  verbose=True):

    global data_lookup
    
    indicators = ['rsi', 'atr', 'wr', 'vr']
    intervals = ['2', '5', '10', '15', '20', '30', '60']
        
    x_train = np.zeros((batch_count, data_column_count, data_row_count))
    y_train = np.zeros((batch_count, n_classes), dtype=int)
    
    symbols = []

    
    #https://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download
    with open('NASD.csv', newline='') as csvfile:
        r = csv.DictReader(csvfile)
        for row in r:
            symbols.append(row['Symbol'])
            
    class_max = [-50,    -20, -15, -10, -5,  -4, -3, -2, -1, -0.1, 0,    1, 2, 3, 4, 5, 10, 15, 20, 50, 10000]
    class_min = [-10000, -50, -20, -15, -10, -5, -4, -3, -2, -1,   -0.1, 0, 1, 2, 3, 4, 5,  10, 15, 20, 50   ]
    
    if not os.path.exists('StockData/'): os.makedirs('StockData/')

    skips = [] # Skiped due to data errors
    
    nics = {}       
    
    if track_classes:
        if os.path.isfile("skips.csv"):
            with open('skips.csv', newline='\n') as csvfile:
                r = csv.DictReader(csvfile)
                for row in r: skips.append(row['Symbols'])            
        else:
            with open("skips.csv", "w") as text_file:
                text_file.write("""Symbols""" + "\n") 
                
        for z in range(0, n_classes):
            nics[z] = []
            if os.path.isfile("nic-" + str(z) + ".csv"):
                with open("nic-" + str(z) + ".csv", newline='\n') as csvfile:
                    r = csv.DictReader(csvfile)
                    for row in r: nics[z].append(row['Symbols'])
            else:
                with open("nic-" + str(z) + ".csv", "w") as text_file:
                    text_file.write("""Symbols""" + "\n")  
                
    batch_i = 0
    
    while batch_i < batch_count:
        
        tsymbol = symbols[np.random.randint(0, len(symbols))] 
    
        tclass = int(np.random.normal(mu, sigma))
        if tclass >= n_classes: tclass = n_classes-1
        elif tclass < 0: tclass = 0
        
        # Check for stocks to skip
        if track_classes: 
            if tsymbol in skips: continue
            if tsymbol in nics[tclass]: continue
             
        if tsymbol in data_lookup:
            if verbose: print("Loading: " + tsymbol)
            df = data_lookup[tsymbol]
        elif os.path.isfile("StockData/" + tsymbol + ".csv"):
            if verbose: print("Loading: " + tsymbol)
            data_lookup[tsymbol] = pan.read_csv("StockData/" + tsymbol + ".csv", 
                                                sep=',', header=0, index_col=0)
            df = data_lookup[tsymbol]
        else:
            if verbose: print("Downloading: " + tsymbol)
            param = {
            'q': tsymbol, # Stock symbol (ex: "AAPL")
            'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
            'x': "NASD", # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': "5Y" # Period (Ex: "1Y" = 1 year)
            }       
            # get price data (return pandas dataframe)
            df = get_price_data(param)                
            df.to_csv("StockData/" + tsymbol + ".csv")
            data_lookup[tsymbol] = deepcopy(df)
        
        tshape = df.iloc[:,0:].values.shape
    
        if tshape[0] <= data_row_count: 
            if verbose: 
                print("Data error: " + tsymbol + 
                      ", continuing to next symbol...")
            if track_classes:
                with open("skips.csv", "a") as text_file:
                        text_file.write(tsymbol + "\n")                   
            continue
    
        if tshape[0] < 400: 
            if verbose: 
                print("Not enough data for: " + tsymbol +
                      ", continuing to next symbol...")
            if track_classes:
                with open("skips.csv", "a") as text_file:
                        text_file.write(tsymbol + "\n")
            continue
        
        # Check for zeros
        if check_for_zeros:
            zero_flag = False
            for row in range(0, tshape[0]):
                for column in range(0, tshape[1]):
                        v = df.iloc[:,column:].values[row][0]
                        if v <= 0:
                            zero_flag = True
                            break
                if zero_flag: break
            if zero_flag: 
                if verbose: 
                    print("Zeros in: " + tsymbol + 
                          ", continuing to next symbol...")
                if track_classes:
                    with open("skips.csv", "a") as text_file:
                            text_file.write(tsymbol + "\n")               
                continue
        
        sdfdata = sdf.retype(df)
                 
        indicies = find_intervals(projection_range, data_row_count + projection_range, 
                                  sdfdata, class_min[tclass], class_max[tclass])
    
        if len(indicies) > 0:
            i = np.random.randint(0, len(indicies))
            if batch_i < batch_count: 
                # Add data to the batch array
                x_train[batch_i] = get_indicator_data_at_index(indicies[i]-projection_range, 
                                   data_row_count, sdfdata, indicators, intervals)
                y_train[batch_i][tclass] = 1
                batch_i += 1
            else:
                break                  
        elif track_classes:
            with open("nic-" + str(tclass) + ".csv", "a") as text_file:
                text_file.write(tsymbol + "\n")  
                
        del sdfdata
        del df
        
    del skips
    del nics

    return x_train, y_train