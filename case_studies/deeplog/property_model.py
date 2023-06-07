from collections.abc import Iterable
from abc import ABC, abstractmethod, abstractproperty

from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet

from pythautomata.abstract.boolean_model import BooleanModel

class PropertyModelAbnormalSymbol(BooleanModel):

    def __init__(self, alphabet : Alphabet, verbose = False):
        self._alphabet = alphabet
        self._abnormal_keys = set([6, 7, 9, 11, 12, 13, 14, 18, 23, 26, 28])
        self._verbose = verbose

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    @property    
    def name(self) -> str:
        return "PropertyModelAbnormalSymbol"

    def accepts(self, sequence: Sequence) -> bool:
        seq = [int(str(symbol)) for symbol in sequence]
        sequence_set = set(seq)
        difference = sequence_set - self._abnormal_keys
        res = len(difference) < len(sequence_set)
        if self._verbose and res: 
            print("Property True:", seq)
        return res


class PropertyModelLessThan(BooleanModel):

    def __init__(self, alphabet : Alphabet, verbose = False):
        self._alphabet = alphabet
        self._verbose = verbose

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    @property    
    def name(self) -> str:
        return "PropertyModelLessThan"

    def accepts(self, sequence: Sequence) -> bool:
        seq = [int(str(symbol)) for symbol in sequence]
        counts = seq.count(4) + seq.count(21)
        res = counts > 5
        if self._verbose and res: 
            print("Property True:", seq)
        return res

