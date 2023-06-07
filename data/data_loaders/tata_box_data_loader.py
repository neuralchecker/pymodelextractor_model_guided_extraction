from lib2to3.pygram import Symbols
import pandas as pd
import numpy as np
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr


class TataBoxDataLoader():
    def __init__(self, data_path):
        self.data_path = data_path             
        self._str_symbol_dict = {x:SymbolStr(x) for x in ["A", "C", "G", "T"]}
        self.alphabet = Alphabet(list(self._str_symbol_dict.values()))
        self._load()        
    
    def _to_sequence(self, list_of_str):
        return Sequence([self._str_symbol_dict[x] for x in list_of_str])


    def _load(self):
        file_name = "/promoters-48+1.txt"        
        #Load files
        with open(self.data_path + file_name, "r") as f:
            data = np.array([ list(line[:-1]) for line in list(f) ])            
            filter = data[:,-1]=='1'
            data = data[filter]
            data = data[:,18:24]
            self.data = list(map(self._to_sequence,data))
