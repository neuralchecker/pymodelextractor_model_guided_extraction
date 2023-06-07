import numpy as np
from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner
from utilities.neural_networks.rnn_language_models.rnn_language_model import RNNLanguageModel
from pythautomata.base_types.symbol import SymbolStr

class TataBoxProbabilityPartitioner(ProbabilityPartitioner):
    def __init__(self, tata_box_model: RNNLanguageModel) -> None:    
        self._model = tata_box_model    
        super().__init__()

    def _get_partition(self, probability_vector):
        A_prob = probability_vector[self._model._get_symbol_index(SymbolStr('A'))]
        C_prob = probability_vector[self._model._get_symbol_index(SymbolStr('C'))]
        G_prob = probability_vector[self._model._get_symbol_index(SymbolStr('G'))]
        T_prob = probability_vector[self._model._get_symbol_index(SymbolStr('T'))]
        if (A_prob + T_prob) >= (C_prob + G_prob):
            return [1]
        return [0]