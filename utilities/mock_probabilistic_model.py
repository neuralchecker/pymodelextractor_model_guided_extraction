

from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
import numpy as np

class MockProbabilisticModel(ProbabilisticModel):
    
    def __init__(self, alphabet:Alphabet, terminal_symbol: Symbol):
        self._alphabet = alphabet
        self._terminal_symbol = terminal_symbol

    
    @property
    def name(self) -> str:
        return "MOCK_PROBABILISTIC_MODEL"
    
    @property
    def terminal_symbol(self) -> Symbol:
        return self._terminal_symbol

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
    
    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    
    def last_token_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
        #    symbols = set(self._alphabet.symbols)
        #    symbols.add(self.terminal_symbol)
        #    return self._get_probability(sequence, symbols, normalize = True)
    
    #TODO: Fix interface, this should be removed from the learners and pymodelextractor as a whole
    def get_last_token_weights(self, sequence, required_suffixes):
        weights = list()
        symbols = list(self.alphabet.symbols) + [self.terminal_symbol]
        
        val =1/len(symbols)
        for _ in required_suffixes:
            weights.append(val)
        return weights  
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        results = []
        for seq in sequences:
            results.append(self.get_last_token_weights(seq, required_suffixes))
        return results
    