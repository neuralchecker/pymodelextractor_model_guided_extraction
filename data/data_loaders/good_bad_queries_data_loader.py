from lib2to3.pygram import Symbols
import pandas as pd
import numpy as np
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr


class GoodBadQueriesDataLoader():
    
    def __init__(self, data_path, size_limit = 200):
        self.data_path = data_path
        alphabet = []
        for i in range(1,256):
            symbol = SymbolStr(str(i))
            alphabet.append(symbol)        
        self.alphabet = Alphabet(alphabet)

        self._load(size_limit)

    def word_to_sequence(self, word):
        seq_list = []
        for char in word:
            symbol = SymbolStr(str(ord(char)+1))
            seq_list.append(symbol)
        return Sequence(seq_list)

    def _load(self, size_limit):
        file_normal = "/goodqueries.txt"
        file_abnormal = "/badqueries.txt"
        #Load files
        with open(self.data_path + file_normal, encoding="ascii", errors="backslashreplace") as f:
            normal = f.read().splitlines()
    
        with open(self.data_path + file_abnormal, encoding="ascii", errors="backslashreplace") as f:
            abnormal = f.read().splitlines()
        
        normal = list(filter(lambda x: len(x) <= size_limit, normal))
        abnormal = list(filter(lambda x: len(x) <= size_limit, abnormal))

        normal = [self.word_to_sequence(query) for query in normal]
        abnormal = [self.word_to_sequence(query) for query in abnormal]        
        self.data = {'good':normal, 'bad':abnormal}
