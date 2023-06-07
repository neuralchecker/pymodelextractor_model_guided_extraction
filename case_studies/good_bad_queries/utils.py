#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import math

from sklearn.model_selection import train_test_split

import random

from numpy.random import seed

pad_value = "pre"
pad_index = 0
vocab_size = 256 + 1

def load_files(folder, file_normal, file_abnormal):
    # normal
    with open(folder + file_normal, encoding="ascii", errors="backslashreplace") as f:
        normal = f.read().splitlines()
    # abnormal
    with open(folder + file_abnormal, encoding="ascii", errors="backslashreplace") as f:
        abnormal = f.read().splitlines()
        
    return normal, abnormal

def get_max_length(normal, abnormal):
    max_length = 0
    # normal
    for query in normal:
        query_len = len(query)
        if query_len > max_length:
            max_length = query_len
    # abnormal
    for query in abnormal:
        query_len = len(query)
        if query_len > max_length:
            max_length = query_len
            
    return max_length

def int_to_char(i):
    if i > 0:
        return chr(i - 1)
    else:
        return pad_value
    
def ints_to_word(ints):
    word = ''
    for i in ints:
        word += int_to_char(i)
    return word

def word_to_ints(word):
    word_ints = []
    for char in word:
        word_ints.append(ord(char) + 1)
    return word_ints

def all_words_to_ints(input_data):
    input_ints = []

    for query in input_data:
        query_ints = word_to_ints(query)
        input_ints.append(query_ints)
        
    return input_ints
    
def get_prefixes(sequence):
    result = list()
    for i in range(1, len(sequence) + 1):
        result.append(sequence[:i])
    return result

def get_windows(queries_dev, window_size=10):
    windows = []
    
    window_size = window_size
    for sequence in queries_dev:                   
        for pref in get_prefixes(sequence): 
            windows.append(pref) 
    
    return windows

from tensorflow.keras.models import load_model, model_from_json

def load_model(exp_name, custom_objects):
    json_model = exp_name + ".json"
    h5_model = exp_name + ".h5"
    json_file = open(json_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    loaded_model.load_weights(h5_model)
    return loaded_model

def get_positions(prob_vector):
    letter_i_prob = {}
    i = 0
    for value in prob_vector:
        letter_i_prob[i] = value
        i += 1        
        
    top = {k: v for k, v in sorted(letter_i_prob.items(), key=lambda item: item[1], reverse=True)}

    positions = {}
    sorted_by_pos = []
    pos = 1
    for key in top:
        prob = top[key]
        positions[key] = pos
        sorted_by_pos.append((key, int_to_char(key), prob))
        pos += 1

    return positions, sorted_by_pos

def print_top_k(sorted_by_pos, k):
    print('Top ', k)
    for i in range(0, k):
        print('Position ', i + 1, ': ', sorted_by_pos[i])    

def softmax(x, temp = 1.):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def probability_of_a_word(word_ints, model, max_length, must_print=False):    
    sequences = []
    for i in range(0, len(word_ints)-1):
        subsequence = word_ints[:i+1]
        sequences.append(subsequence)
    
    encoded = pad_sequences(sequences, maxlen=max_length-1, padding=pad_value, value=pad_index)
        
    yhat = model.predict(encoded, verbose=0)
    
    ranking = []
    result_prod = 1
    result_log = 0
    for i in range(len(word_ints)-1):
        letter_int = word_ints[i+1]
        letter = int_to_char(letter_int)
        if must_print:
            print('Probability of: ', letter, ' (', letter_int, ')')
            print('Having: ', ints_to_word(sequences[i]), ' (', sequences[i], ')')
            print('Is: ', yhat[i][-1][letter_int])
        
        positions, sorted_by_pos = get_positions(yhat[i][-1])
        
        ranking.append(positions[letter_int])
        
        if must_print:
            print('Ranking: ', positions[letter_int])        
            print_top_k(sorted_by_pos, 5)
            print('#########################')
            
        result_prod = result_prod * yhat[i][-1][letter_int]
        if yhat[i][-1][letter_int] > 0:
            result_log = result_log + math.log(yhat[i][-1][letter_int])
    return result_log, result_prod, ranking

def probability_of_a_word_without_token(word_ints, model, max_length, must_print=False):
    sequences = []
    for i in range(0, len(word_ints)-1):
        subsequence = word_ints[:i+1]
        sequences.append(subsequence)
       
    encoded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length-1, padding=pad_value, value=pad_index)
        
    yhat = model.predict(encoded, verbose=0)
    
    ranking = []
    result_prod = 1
    result_log = 0
    for i in range(len(word_ints)-1):
        prob_vector = softmax(yhat[i][-1][1:])
        prob_vector = np.insert(prob_vector, 0, 0) # to keep positions
        
        letter_int = word_ints[i+1]
        letter = int_to_char(letter_int)
        if must_print:
            print('Probability of: ', letter, ' (', letter_int, ')')
            print('Having: ', ints_to_word(sequences[i]), ' (', sequences[i], ')')
            print('Is: ', prob_vector[letter_int])
        
        positions, sorted_by_pos = get_positions(prob_vector)
        
        ranking.append(positions[letter_int])
        
        if must_print:
            print('Ranking: ', positions[letter_int])        
            print_top_k(sorted_by_pos, 5)
            print('#########################')
            
        result_prod = result_prod * prob_vector[letter_int]
        if prob_vector[letter_int] > 0:
            result_log = result_log + math.log(prob_vector[letter_int])
    return result_log, result_prod, ranking

def print_probability_of_a_word(word):
    word_ints = word_to_ints(word)
    return probability_of_a_word(word_ints, must_print=True)

def print_probability_of_a_word_without_token(word):
    word_ints = word_to_ints(word)
    return probability_of_a_word_without_token(word_ints, must_print=True)

def choose_best_threshold(tpr, fpr, thresholds):
    #Distances to (0,1) (minimum distance may not be the optimal but is easy to compute)
    dist = np.sqrt((fpr-0)**2 + (tpr-1)**2)
    min_dist_index = np.argmin(dist)
    best_threshold = thresholds[min_dist_index]
    return best_threshold, min_dist_index

import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

def my_classification_report(test_pred, test_truth):
    print("Classification report:")   
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = classification_report(test_truth, test_pred)
        print(report)

    print(" ")
    print("Confusion matrix: ") 
    matrix = confusion_matrix(test_truth, test_pred)
    print(matrix)
    
    print(" ")
    accu = accuracy_score(test_truth, test_pred)
    print("Accuracy: ", accu) 
    
    print(" ")
    recall = recall_score(test_truth, test_pred)
    print("Recall: ", recall)
    
    return report, matrix, accu, recall

def get_next_symb_probas(word):
    probas = []
    for i in range(0,len(word)-2) :
        _, p, _ = probability_of_a_word_without_token(word_to_ints(word[i:i+2]), False)
        probas.append(p)
    # return min(probas), max(probas)
    return 0, max(probas)

def get_sequences(word_ints, maxlen=0):
    sequences = []
    for i in range(0, len(word_ints)):
        subsequence = word_ints[max(0,i+1-maxlen):i+1]
        sequences.append(subsequence)
    return sequences

def get_pad_sequences(word_ints, maxlen=0):
    sequences = []
    for i in range(0, len(word_ints)):
        word = word_ints[max(0,i+1-maxlen):i+1]
        subsequence = np.zeros(maxlen, dtype=int)
        subsequence[-len(word):] = word
        sequences.append(subsequence)
    return np.array(sequences)

def get_pad_sequences_from_words(words, maxlen=0):
    pad_sequences = []
    for w in words:
        subsequences = get_pad_sequences(w, maxlen=maxlen)
        pad_sequences += subsequences.tolist()
    return np.array(pad_sequences)

def get_prediction(model, word_ints, window_size, verbose=False):
    
    # preprocess
    
    # sequences = get_sequences(word_ints)
    # encoded = pad_sequences(sequences, maxlen=window_size-1, padding=pad_value, value=pad_index)
    
    encoded = get_pad_sequences(word_ints, maxlen=window_size-1)
    
    # predict
    
    yhat = model.predict(encoded)

    # initialize stat variables
    
    prev_hit = False
    consecutive_hits = 0
    # nhits = 0
    consecutive_hits_list = []
    yerror_p = 0.
    # min_proba = 1.
    # max_proba = 0.
    cum_hits_proba = 0.
    log_proba = 0.
    
    starting_hit = None
    
    # process sequences
    
    for i in range(len(encoded)-1):

        ytrue = encoded[i+1][-1]
        ytrue_p  = yhat[i][-1][ytrue]
        ypred = np.argmax(yhat[i][-1])
        ypred_p  = np.max(yhat[i][-1])
        
        yerror_p += abs(ytrue_p - ypred_p)
       
        log_proba += np.log(ytrue_p)
        
        hit = (ytrue == ypred)       
        # nhits += int(hit)
        
        if prev_hit: 
            consecutive_hits += int(hit)
            cum_hits_proba += ytrue_p if hit else 0
        else:
            starting_hit = i+1 if hit else None 
            consecutive_hits = int(hit)
            cum_hits_proba = ytrue_p if hit else 0

        if prev_hit and not hit:
            consecutive_hits_list.append((starting_hit, consecutive_hits, cum_hits_proba))
            
        prev_hit = hit
            
        if verbose:
            print("window: ", i, "\thit: ", hit,
                  "\tytrue: ", ytrue, "\tytrue_p: ", ytrue_p, "\typred: ", ypred, "\typred_p: ", ypred_p)

    if starting_hit is not None:
            consecutive_hits_list.append((starting_hit, consecutive_hits, cum_hits_proba))
        
    return consecutive_hits_list, yerror_p, log_proba, len(encoded)

import sys

def compute_predictions(model, data, window_size, verbose=False):
    data_predictions = []
    for i in range(len(data)):
        sys.stdout.write("Progress: %d  \r" % (i) )
        sys.stdout.flush()
        word_ints = word_to_ints(data[i])
        prediction = get_prediction(model, word_ints, window_size=window_size, verbose=verbose)
        data_predictions.append(prediction)
    return data_predictions


    
    