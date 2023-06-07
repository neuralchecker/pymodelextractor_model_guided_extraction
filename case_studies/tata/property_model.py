from collections.abc import Iterable
from abc import ABC, abstractmethod, abstractproperty

from base_types.sequence import Sequence
from base_types.alphabet import Alphabet

from abstractions.queryable_model import QueryableModel

class PropertyModel(QueryableModel):

    def __init__(self, alphabet : Alphabet, min_pos = 0, tata_seq_length = 6, verbose = True):
        self._alphabet = alphabet
        self._tata_seq_length = tata_seq_length
        self._min_pos = min_pos
        self._verbose = verbose

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    def accepts(self, sequence: Sequence) -> bool:
        sequence_str = str(sequence)
        countTA = sequence_str.count("T", self._min_pos, self._tata_seq_length) + sequence_str.count("A", self._min_pos, self._tata_seq_length)
        countCG = sequence_str.count("C", self._min_pos, self._tata_seq_length) + sequence_str.count("G", self._min_pos, self._tata_seq_length)
        res = countTA < countCG
        if self._verbose and res: 
            print("Property True", sequence_str)
        return res    




