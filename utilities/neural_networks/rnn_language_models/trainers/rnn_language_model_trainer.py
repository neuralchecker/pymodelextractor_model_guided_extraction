#from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

import math
import numpy as np
from typing import List
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence

from utilities.neural_networks.rnn_language_models.rnn_language_model import RNNLanguageModel

from utilities.symbol_encoder import SymbolEncoder
from utilities.neural_networks.model_definitions import Models

from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from abc import ABC, abstractmethod, abstractproperty


class RNNLanguageModelTrainer(ABC):

    def __init__(self, model_output_path, data: List[Sequence], window_size: int, alphabet: Alphabet, padding_symbol, terminal_symbol, seed = 42, params = None, model = None, has_embedding = True):
        self.model_output_path = model_output_path
        self.data = data
        self.window_size = window_size
        self.alphabet = alphabet
        self.padding_symbol = padding_symbol
        self.terminal_symbol = terminal_symbol
        self.seed = seed
        self._use_one_hot_on_input = not has_embedding
        if params == None:
            self.params = self._get_default_params()
        else:
            self.params = params
        if model == None:
            self.model = self._get_default_model()
        else:
            self.model = model
        self.compile_model()

    @abstractmethod
    def compile_model(self):
        raise NotImplementedError
        

    @abstractmethod
    def generate_target(self, transformed_data, encoder):
        raise NotImplementedError
    
    
    def generate_encoder(self):
        encoder_alphabet = set(self.alphabet.symbols)
        #encoder_alphabet.add(self.padding_symbol)
        #encoder_alphabet.add(self.terminal_symbol)
        encoder_alphabet = Alphabet(frozenset(encoder_alphabet))
        encoder = SymbolEncoder(encoder_alphabet, terminal_symbol= self.terminal_symbol, padding_symbol= self.padding_symbol)
        return encoder
            
    @abstractmethod
    def _get_default_model(self):
        raise NotImplementedError
        
    @abstractmethod
    def instantiate_language_model(self, encoder, model):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate_model(self, model, x_test, y_test):   
        raise NotImplementedError

    def train_network(self):
        # Every sequence in data is assumed to belong to the target language
        # Every sequence in data does not end with terminal_symbol, it should be added (if given)
        working_data = self.data
        #Add terminal symbol to data 
        if self.terminal_symbol is not None: working_data =  list(map(lambda x: x.append(self.terminal_symbol), self.data))   

        #Make windowing
        sequences = []
        for sequence in working_data:                   
            for pref in sequence.get_prefixes(): 
                sequences.append(pref)    
            #if len(sequence) < self.window_size:
            #    sequences.append(sequence)
            #for i in range(0, len(sequence)- self.window_size+1): 
            #    subsequence = Sequence(sequence[i:i+self.window_size])                 
            #    sequences.append(subsequence)    
        
        #Add padding (this may be tidied up)
        generator = UniformLengthSequenceGenerator(self.alphabet, self.window_size, self.seed)
        padded_data = list(map(lambda x: generator.pad(x,self.padding_symbol,self.window_size,'pre'), sequences))

        ###THIS DEPENDS IF PADDING IS CONTEMPLATED
        encoder = self.generate_encoder()
        
        transformed_data = list(
            map(lambda x: list(map(lambda y: encoder.encode(y), x)), padded_data))
        transformed_data = np.asarray(transformed_data)
        
        #Generate Target
        X, y = self.generate_target(transformed_data, encoder)    

        x_dev, x_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.seed)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_dev, y_dev, test_size=0.33, random_state=self.seed)

        # Get Model        
        model = self.model
       
        # Train Model   
        patience = self.params['patience']
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        earlystopper = EarlyStopping(
            monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
        
        model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid),
                  batch_size=batch_size, callbacks=[earlystopper])
        
        language_model = self.instantiate_language_model(encoder, model)
        language_model.persist()

        # Evaluate model and persist evalulation metrics
        result_dict = self.evaluate_model(model, x_test, y_test)
        self.results_to_file(result_dict)
        return language_model, result_dict

  
    def _get_default_params(self):
        return{'patience':5, 'epochs':10, 'batch_size':100, 'learning_rate':0.01}

    def results_to_file(self, result_dict):
        file = open(self.model_output_path + "/test_results.txt", "w+")
        for key in result_dict.keys():
            file.write(key+": " + str(result_dict[key]))
        
        file.flush()
        file.close()
    
    def _get_vocab_size(self):
        vocab_size = len(self.alphabet.symbols) + 1
        if self. terminal_symbol is not None:
            vocab_size = vocab_size + 1
        return vocab_size


    