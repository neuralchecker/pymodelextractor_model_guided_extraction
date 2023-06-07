import numpy as np
from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner
from utilities.neural_networks.rnn_language_models.rnn_language_model import RNNLanguageModel
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner
from case_studies.good_bad_queries.good_bad_quereies_language_model import GoodBadQueriesLanguageModel

class GoodBadQueriesHasAttackSymbolProbabilityPartitioner(ProbabilityPartitioner):
    def __init__(self, topk: int,  good_bad_queries_model: GoodBadQueriesLanguageModel) -> None:    
        self._topk = topk    
        self._model = good_bad_queries_model
        self._inner_partitioner = TopKProbabilityPartitioner(topk)
        super().__init__()

    def _get_partition(self, probability_vector):
        lesser_symbol = self._model._get_symbol_index(SymbolStr(str(ord('<'))))
        greater_symbol = self._model._get_symbol_index(SymbolStr(str(ord('>'))))
        semicolon_symbol = self._model._get_symbol_index(SymbolStr(str(ord(';'))))
        hashtag_symbol = self._model._get_symbol_index(SymbolStr(str(ord('#'))))
        topk = self._inner_partitioner.get_partition(probability_vector)
        for pos in [lesser_symbol, greater_symbol, semicolon_symbol, hashtag_symbol]:
            if topk[pos]==1:
                return [1]
        return [0]
        