import pandas as pd
import numpy as np
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr

class SpiceDataLoader():
    
    def __init__(self, data_path):
        self.data_path = data_path
        self._load()

    def _load(self):
        #Load file
        df = pd.read_csv(self.data_path)
        df.columns = ['raw_seq']
        
        #Convert rows to np arrays
        df['arr'] = df['raw_seq'].apply(lambda x: self._preprocess_line(x))
        
        #Find minimum and substract it from every sequence to make the minimum 0, then add 1.
        #This is done for zero-padding compatibility
        df['min'] = df['arr'].apply(lambda x: np.min(x))
        min_value = df['min'].min()
        df['arr'] = df['arr'].apply(lambda x: x - min_value + 1)

        #data is now a list of np arrays
        data = list(df['arr'])

        #Make data a list of Sequences
        data = list(map(lambda x: Sequence([SymbolStr(symbol) for symbol in x]), data))
        
        self.data = data
        
        #Define the alphabet
        self.alphabet = self._get_alphabet(data)

    def _preprocess_line(self,line):
        result = np.array(line.split(), dtype = int) 
        #the first element is discarded as it is the length of the sequence       
        return result[1:]
    
    def _get_alphabet(self, data):
        symbol_set = set()
        for seq in data:
            for symbol in seq:
                symbol_set.add(symbol)
        
        return Alphabet(frozenset(symbol_set))