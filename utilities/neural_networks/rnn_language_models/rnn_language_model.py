from collections.abc import Iterable
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy.special import softmax
from typing import List
from collections import defaultdict

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
from abc import ABC, abstractmethod, abstractproperty
from pythautomata.abstract.probabilistic_model import ProbabilisticModel

#Class that allows to persist, load and query Keras recurrent neural networks (language models)
#Assumptions:
#-Not fixed sequence length
#-There is an ending symbol
#-Network was trained with a framework trainer (for example RNNLanguageModelRecurrentNeuralNetworkTrainer)
class RNNLanguageModel(ProbabilisticModel, ABC):
    def __init__(self, directory, alphabet: Alphabet = None, enconder: SymbolEncoder = None,
                 model=None, training_seq_length=None, padding_symbol: SymbolStr = None, terminal_symbol: SymbolStr = None, use_one_hot_on_input = False, name = None):
        self._directory = directory
        self._alphabet = alphabet
        self._encoder = enconder
        self._model = model
        self._training_seq_length = training_seq_length
        self._padding_symbol = padding_symbol
        self._terminal_symbol = terminal_symbol   
        self._use_one_hot_on_input = use_one_hot_on_input   
        self._name = name                
        
        if alphabet is None or enconder is None or model is None or training_seq_length is None or padding_symbol is None or terminal_symbol is None:
            try:
                self.load()
                self.loaded = True
                print('The model has been successfully loaded')
            except:
                tb = traceback.format_exc()
                self.loaded = False
                print("One of the files failed to load")
            #print(tb)
        self._vocab_size = len(self._alphabet) + 2 if self._alphabet is not None else None
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    @property
    def terminal_symbol(self):
        return self._terminal_symbol
    
    @abstractmethod
    def _to_categorical(self, sequence):
        raise NotImplementedError

    @abstractmethod
    def sequence_probability(self, sequence: Sequence, debug = False):
        raise NotImplementedError

    @abstractmethod
    def raw_next_symbol_probas(self, sequence: Sequence):
        raise NotImplementedError
    
    @abstractmethod
    def _get_symbol_index(self, symbol):
        raise NotImplementedError
    
    @abstractmethod
    def _get_encoded_symbol_index(self, encoded_symbol):
        raise NotImplementedError
    
    @abstractmethod
    def next_symbol_probas(self, sequence: Sequence):
        raise NotImplementedError

    def raw_eval(self, sequence: Sequence):
        if not hasattr(self, '_model'):
            raise AttributeError     
        
        generator = UniformLengthSequenceGenerator(self.alphabet, self._training_seq_length)
        if len(sequence) == 0:
            sequence = generator.pad(sequence, self._padding_symbol, padding_type='pre')

        encoded_sequence = np.asarray(self._encoder.encode_sequence(sequence))
        if self._use_one_hot_on_input:
            encoded_sequence = self._to_categorical(encoded_sequence)
            encoded_sequence = np.reshape(encoded_sequence, (1, len(encoded_sequence),-1))            
        else:
            encoded_sequence = np.reshape(encoded_sequence, (1, len(encoded_sequence)))

        model_evaluation = self._model.predict(encoded_sequence)
        return model_evaluation

    #TODO test
    def raw_eval_batch(self, sequences: List[Sequence]):
        if not hasattr(self, '_model'):
            raise AttributeError

        sequences_by_length = defaultdict(lambda: [])
        for seq in sequences:
            sequences_by_length[len(seq)].append(seq)        
        if len(sequences_by_length[0]) > 0:  
            generator = UniformLengthSequenceGenerator(self.alphabet, self._training_seq_length)
            sequences_by_length[0] = generator.pad_sequences(sequences_by_length[0], self._padding_symbol, padding_type='pre')
        else:
            #the if statement generates a key with no elements
            sequences_by_length.pop(0, None)        
        query_results = []
        seqs_to_query = []
        for length in sequences_by_length:            
            seqs =  sequences_by_length[length]
            encoded_sequences = list(map(lambda x: self._encoder.encode_sequence(x), seqs))        
            if self._use_one_hot_on_input:
                encoded_sequences = self._to_categorical(encoded_sequences)
            
            encoded_sequences_np = np.asarray(encoded_sequences)

            if length == 1:
                encoded_sequences_np = encoded_sequences_np.reshape((-1, 1, len(encoded_sequences_np[0]))) 
            if length == 0:                
                seqs = [Sequence() for i in seqs]

            model_evaluation = self._model.predict(encoded_sequences_np)
            seqs_to_query.extend(seqs)
            query_results.extend(model_evaluation)

        result_dict = dict(zip(seqs_to_query, query_results))            
        return result_dict

    
    def persist(self):
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)        
        # Persisting model
        self._model.save(self._directory + '/keras_model' + '.h5')
        elements_to_persist = {
            'alphabet': self._alphabet,
            'encoder': self._encoder,
            'training_seq_length': self._training_seq_length,
            'padding_symbol': self._padding_symbol,
            'terminal_symbol': self._terminal_symbol,
            'use_one_hot': self._use_one_hot_on_input
        }

        for key in elements_to_persist:
            if elements_to_persist[key] is None:
                raise Exception(f"Atrribute: {key} is None")

            joblib.dump(elements_to_persist[key],
                        f"{self._directory}/{key}.joblib")

    def load(self):
        self._model = load_model(self._directory + 'keras_model' + '.h5')
        self._alphabet = joblib.load(self._directory + 'alphabet' + '.joblib')
        self._encoder = joblib.load(self._directory + 'encoder' + '.joblib')
        self._training_seq_length = joblib.load(self._directory + 'training_seq_length' + '.joblib')
        self._padding_symbol = joblib.load(self._directory + 'padding_symbol' + '.joblib')
        self._terminal_symbol = joblib.load(self._directory + 'terminal_symbol' + '.joblib')
        self._use_one_hot_on_input = joblib.load(self._directory+ 'use_one_hot' + '.joblib')

    
    def sequence_weight(self, sequence):
        return self.sequence_probability(sequence)[0]

    def log_sequence_weight(self, sequence):
        return self.sequence_probability(sequence)[1]

    def get_last_token_weights(self, sequence, required_suffixes):
        weights = list()
        alphabet_symbols_weights = self.next_symbol_probas(sequence)
        alphabet_symbols_weights = {Sequence() + k: alphabet_symbols_weights[k] for k in alphabet_symbols_weights.keys()}
        for suffix in required_suffixes:
            if suffix in alphabet_symbols_weights:
                weights.append(alphabet_symbols_weights[suffix])
            else:
                new_sequence = sequence + suffix
                new_prefix = Sequence(new_sequence[:-1])
                new_suffix = new_sequence[-1]
                next_symbol_weights = self.next_symbol_probas(new_prefix)
                weights.append(next_symbol_weights[new_suffix])
        return weights

    def get_last_token_weights_batch(self, sequences, required_suffixes):
        seqs_to_query = set()
        symbols = list(self.alphabet.symbols) + [self._terminal_symbol]
        for seq in sequences:
            for required_suffix in required_suffixes:
                if required_suffix not in symbols and len(required_suffix)>1:
                    seqs_to_query.add(seq+required_suffix[:-1])
                else:
                    seqs_to_query.add(seq)

        result_dict = self.raw_eval_batch(list(seqs_to_query))
        #result_dict = dict(zip(seqs_to_query, query_results))
        results = []
        for seq in sequences:
            seq_result = []
            for required_suffix in required_suffixes:
                if required_suffix not in symbols and len(required_suffix)>1:
                    seq_result.append(result_dict[seq+required_suffix[:-1]][required_suffix[-1]])
                else:
                    if required_suffix not in symbols:
                        required_suffix = SymbolStr(required_suffix.value[0].value)
                    seq_result.append(result_dict[seq][self._get_symbol_index(required_suffix)])
            results.append(seq_result)
        
        return results
