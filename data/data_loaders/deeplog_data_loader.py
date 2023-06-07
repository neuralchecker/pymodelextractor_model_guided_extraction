from lib2to3.pygram import Symbols
import pandas as pd
import numpy as np
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr


class DeepLogDataLoader():
    
    def __init__(self, data_path):
        self.data_path = data_path
        self._load()

    def _load(self):
        #Load file
        df = pd.read_csv(self.data_path, header = None)
        df.columns = ['raw_seq']
        
        #Convert rows to np arrays
        df['arr'] = df['raw_seq'].apply(lambda x: self._preprocess_line(x))        
        
        #data is now a list of np arrays
        data = list(df['arr'])

        #Make data a list of Sequences
        data = list(map(lambda x: Sequence([SymbolStr(symbol) for symbol in x]), data))
        
        self.data = data
        
        #Define the alphabet
        self.alphabet = self._get_alphabet(data)

    def _preprocess_line(self,line):
        result = np.array(line.split(), dtype = int)        
        return result
    
    def _get_alphabet(self, data):
        symbol_set = set()
        #Symbols in deelog go from 1 to 29
        for i in range(1,30):
            symbol_set.add(SymbolStr(i))
        return Alphabet(frozenset(symbol_set))