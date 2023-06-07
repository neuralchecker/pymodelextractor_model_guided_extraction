from collections.abc import Iterable
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy.special import softmax

import joblib
import numpy as np
import os
import traceback

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr


from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from utilities.symbol_encoder import SymbolEncoder
from utilities.neural_networks.rnn_language_models.rnn_language_model import RNNLanguageModel

#Class that allows to persist,load and query Keras recurrent neural networks (language models)
#Assumptions:
#-Not fixed sequence length
#-There is an ending symbol
#-Network was trained with a framework trainer (for example RNNLanguageModelTrainer)
class TimeDistributedLanguageModel(RNNLanguageModel):
    def raw_next_symbol_probas(self, sequence: Sequence):
        result = self.raw_eval(sequence)
        next_probas = result[-1][0]
        return next_probas   
    
    
    def _to_categorical(self, sequence):
        return to_categorical(sequence, self._vocab_size)

    def sequence_probability(self, sequence: Sequence, debug = False):
        prefixes = []
        prefixes.append(Sequence([]))
        for i in range(0, len(sequence)-1): 
            prefix = sequence.value[:i+1] 
            prefixes.append(Sequence(prefix)) 
            
        if debug: print('Prefixes', prefixes)  
        
        generator = UniformLengthSequenceGenerator(self.alphabet, self._training_seq_length)
        padded_sequences = generator.pad_sequences(prefixes, padding_type='pre', padding_symbol=self._padding_symbol)

        if debug: print('Padded:',padded_sequences)

        encoded_sequences = list(map(lambda x: self._encoder.encode_sequence(x), padded_sequences))        
        encoded_sequences_np = np.asarray(encoded_sequences)
        
        if debug: print('Encoded:', encoded_sequences_np)
                 
        yhat = self._model.predict(encoded_sequences_np, verbose=0)
        
        result = 1
        log_result = 0

        word = np.asarray(self._encoder.encode_sequence(sequence))
        for i in range(len(word)):
            letter = word[i]
            symbol_index = self._get_encoded_symbol_index(letter)   
            
            if debug: 
                print('Letter: ', self._encoder.decode(letter), '-', letter)
                print('Symbol index: ',symbol_index)
                print('Pred: ',yhat[i])
                print('Value in letter: ', yhat[i][-1][symbol_index])
            result = result * yhat[i][-1][symbol_index]    
            log_result = log_result + np.log(yhat[i][-1][symbol_index])    
        return result, log_result

    def _get_symbol_index(self, symbol):
        return np.argmax(to_categorical(self._encoder.encode(symbol),num_classes=len(self.alphabet)+2))     

    def _get_encoded_symbol_index(self, encoded_symbol):
        return np.argmax(to_categorical(encoded_symbol,num_classes=len(self.alphabet)+2))    

    def next_symbol_probas(self, sequence: Sequence):
        """
        Function that returns a dictionary with the probability of next symbols (not including padding_symbol)
        Quickly implemented, depends on raw_next_symbol_probas(sequence) and uses a softmax over some symbols 
        of the network output as the sum is not one (eg. time distributed case).
        """                
        next_probas = self.raw_next_symbol_probas(sequence)

        symbols = list(self.alphabet.symbols) + [self._terminal_symbol]
        intermediate_dict = {}
        probas = np.zeros(len(symbols))
        for idx, symbol in enumerate(symbols):
            proba = next_probas[self._get_symbol_index(symbol)]
            intermediate_dict[symbol] = (proba, idx)
            probas[idx] = proba

        probas = softmax(probas)       

        dict_result = {}
        for symbol in intermediate_dict.keys():
            dict_result[symbol] = probas[intermediate_dict[symbol][1]]

        return dict_result        

    def last_token_probabilities(self, sequence: Sequence):
        return self.next_symbol_probas(sequence)       

    def last_token_probability(self, sequence: Sequence):
        return self.next_symbol_probas(sequence[:-1])[sequence[-1]]

    def log_sequence_probability(self, sequence: Sequence):
        raise NotImplementedError