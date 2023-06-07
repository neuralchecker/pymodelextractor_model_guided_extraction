import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from pythautomata.abstract.boolean_model import BooleanModel
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner

from utilities.neural_networks.rnn_language_models.rnn_language_model import RNNLanguageModel

class DeepLogTopKModel(BooleanModel):

    def __init__(self, model: RNNLanguageModel, k: int):
        super().__init__()
        self.model = model
        self.k = k
        self.partitioner = TopKProbabilityPartitioner(k)
        self.query_cache = dict()

    @property
    def model_name(self) -> str:
        return self.model.name

    @property
    def alphabet(self):
        return self._alphabet

    @property    
    def name(self) -> str:
        return "DeepLogTopKModel_"+self.model_name

    def _next_symbol_probas(self, sequence):
        if sequence not in self.query_cache:
            self.query_cache[sequence] = self.model.next_symbol_probas(sequence)
        return self.query_cache[sequence] 
        

    def accepts(self, sequence: Sequence) -> bool:
        prefixes = sequence.get_prefixes()
        prefixes_next_probas = list()
        for prefix in prefixes:
            prefixes_next_probas.append(self._next_symbol_probas(prefix))
        for i in range(len(sequence)):
            next_probas = list(prefixes_next_probas[i].values())
            index_of_symbol = self.model._get_symbol_index(sequence[i])
            if self.partitioner.get_partition(next_probas)[index_of_symbol]!=1:
                return False
        return True

