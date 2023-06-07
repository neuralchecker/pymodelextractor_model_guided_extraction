import numpy as np

from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner

from case_studies.good_bad_queries.utils import load_model
from case_studies.good_bad_queries.transformers import get_custom_objects

class SyncronicExcecutionLanguageModel(ProbabilisticModel):

    def __init__(self, model1:ProbabilisticModel, model2:ProbabilisticModel, probability_partitioner: ProbabilityPartitioner,max_seq_length: int = None,  model_name: str = None, compose_by_difference = False):
        super().__init__()
        if model_name is None:
            self._model_name = model1.name+"_"+model2.name
        else:
            self._model_name = model_name
        if max_seq_length is None:
            self._max_seq_length = min(model1._max_seq_length, model2._max_seq_length)
        self._model1 = model1
        self._model2 = model2
        self.query_cache = dict()
        assert model1.alphabet == model2.alphabet
        assert model1._padding_symbol == model2._padding_symbol
        assert model1.terminal_symbol == model2.terminal_symbol

        self._alphabet = model1.alphabet
        self._padding_symbol = model1._padding_symbol
        self._terminal_symbol = model1.terminal_symbol
        self._compose_by_difference = compose_by_difference
        self._partitioner = probability_partitioner

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alphabet(self):
        return self._alphabet

    @property    
    def name(self) -> str:
        return self.model_name

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
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        model1_results = self._model1.get_last_token_weights_batch(sequences, required_suffixes)
        model2_results = self._model2.get_last_token_weights_batch(sequences, required_suffixes)
        res = []
        for i in range(len(model1_results)):
            res1 = model1_results[i]
            res2 = model2_results[i]
            res.append(self._compose_probas(res1, res2))
        return res

    def _compose_probas(self, probability_vector1, probability_vector2):
        exception_vector = np.ones(len(probability_vector1))*-2
        if self._compose_by_difference:
            if not self._partitioner.are_in_same_partition(probability_vector1, probability_vector2):
                return probability_vector1
            return exception_vector
        else:
            if self._partitioner.are_in_same_partition(probability_vector1, probability_vector2):
                return probability_vector1
            return exception_vector
        
    
    def get_last_token_weights(self, sequence, required_suffixes):
        model1_results = self._model1.get_last_token_weights(sequence, required_suffixes)
        model2_results = self._model2.get_last_token_weights(sequence, required_suffixes)
        return self._compose_probas(model1_results, model2_results)
    
    def last_token_probabilities_batch(self, sequences: list[Sequence], required_suffixes: list[Sequence]) -> \
            list[list[float]]:
        return self.get_last_token_weights_batch(sequences, required_suffixes)
        