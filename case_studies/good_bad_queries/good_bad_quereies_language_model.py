import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator

from case_studies.good_bad_queries.utils import load_model
from case_studies.good_bad_queries.transformers import get_custom_objects

class GoodBadQueriesLanguageModel(ProbabilisticModel):

    def __init__(self, model_name: str, max_seq_length: int):
        super().__init__()
        path = "./case_studies/good_bad_queries/goodbadqueries-transf-lm/"+model_name
        self._model = load_model(path, get_custom_objects())
        self.query_cache = dict()
        self._model_name = model_name
        alphabet = []
        for i in range(1,256):
            symbol = SymbolStr(str(i))
            alphabet.append(symbol)        
        self._alphabet = Alphabet(alphabet)
        self._max_seq_length = max_seq_length
        self._padding_symbol = SymbolStr(str(0))
        self._terminal_symbol = SymbolStr(str(256))

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alphabet(self):
        return self._alphabet

    @property    
    def name(self) -> str:
        return "TransformerModel_"+self.model_name

    def _next_symbol_probas(self, sequence):
        if sequence not in self.query_cache:
            self.query_cache[sequence] = self.next_symbol_probas(sequence)
        return self.query_cache[sequence] 

    @property            
    def terminal_symbol(self) -> Symbol:
        return self._terminal_symbol

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def last_token_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def _get_symbol_index(self, symbol):
        return int(symbol.value)  

    def raw_eval_batch(self, sequences: list[Sequence], batch_size = 10000):
        if not hasattr(self, '_model'):
            raise AttributeError
        if len(sequences) == 1:
            return {sequences[0]: self.raw_eval(sequences[0])}

        generator = UniformLengthSequenceGenerator(self.alphabet, self._max_seq_length)  

        padded_sequences = generator.pad_sequences(sequences, self._padding_symbol, padding_type='pre')

        encoded_sequences = list(map(lambda x: self.encode_sequence(x), padded_sequences))         
        encoded_sequences_np = np.asarray(encoded_sequences)
        
        query_results = self._model.predict(encoded_sequences_np, batch_size)
    
        result_dict = dict(zip(sequences, query_results))            
        return result_dict

    def last_token_probabilities_batch(self, sequences: list[Sequence], required_suffixes: list[Sequence]) -> \
            list[list[float]]:
        return self.get_last_token_weights_batch(sequences, required_suffixes)


    def get_last_token_weights_batch(self, sequences, required_suffixes):
        seqs_to_query = set()
        symbols = list(self.alphabet.symbols) + [self._terminal_symbol]
        for seq in sequences:
            for required_suffix in required_suffixes:
                if required_suffix not in symbols and len(required_suffix)>1:
                    seqs_to_query.add(seq+required_suffix[:-1])
                else:
                    seqs_to_query.add(seq)

        result_dict = self.raw_eval_batch(seqs_to_query)
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
                    seq_result.append(result_dict[seq][-1][self._get_symbol_index(required_suffix)])
            results.append(seq_result)
        
        return results

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

    def next_symbol_probas(self, sequence: Sequence):
        """
        Function that returns a dictionary with the probability of next symbols (not including padding_symbol)
        Quickly implemented, depends on raw_next_symbol_probas(sequence) 
        """                
        next_probas = self.raw_next_symbol_probas(sequence)[-1]

        symbols = list(self.alphabet.symbols) + [self._terminal_symbol]
        intermediate_dict = {}
        probas = np.zeros(len(symbols))
        for idx, symbol in enumerate(symbols):
            proba = next_probas[self._get_symbol_index(symbol)]
            intermediate_dict[symbol] = (proba, idx)
            probas[idx] = proba       

        dict_result = {}
        for symbol in intermediate_dict.keys():
            dict_result[symbol] = probas[intermediate_dict[symbol][1]]
        assert self._terminal_symbol in dict_result
        return dict_result   
    
    def raw_next_symbol_probas(self, sequence: Sequence):
        result = self.raw_eval(sequence)
        next_probas = result[0]
        return next_probas   
    
    def raw_eval(self, sequence: Sequence):
        if not hasattr(self, '_model'):
            raise AttributeError     
        
        generator = UniformLengthSequenceGenerator(self.alphabet, self._max_seq_length)
        
        sequence = generator.pad(sequence, self._padding_symbol, padding_type='pre')

        encoded_sequence = np.asarray(self.encode_sequence(sequence))
        
        encoded_sequence = np.reshape(encoded_sequence, (1, len(encoded_sequence)))

        model_evaluation = self._model.predict(encoded_sequence)
        return model_evaluation

    def encode_sequence(self, sequence):
        return list(map(lambda x: int(x.value), sequence.value))