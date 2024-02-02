import numpy as np
from random import seed
from random import randint
from random import choices 
from random import choice

from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.base_types.symbol import Symbol, SymbolStr
from pythautomata.base_types.sequence import Sequence

class DataSequenceGenerator(SequenceGenerator):

    def __init__(self, data , alphabet, max_seq_length: int, random_seed: int = 21, min_seq_length: int = 0):        
        super().__init__(alphabet, max_seq_length, random_seed, min_seq_length)
        self.data = data
    
    def generate_single_word(self, length):
        raise NotImplementedError

    def generate_word(self, length):
        if length > self._max_seq_length:
            raise AssertionError("Param length cannot exceed max_seq_length")
        word = choice(self.data)
        return self.cut(word)

    def generate_words(self, number_of_words: int):
        words = choices(self.data, k = number_of_words)
        return self.cut_sequences(words)
    
    def cut(self, word: Sequence, type = 'pre', max_len: int = None):
        if max_len is None:
            max_len = self._max_seq_length
        value = list(word.value)
        if len(value) > max_len:
            if type == 'post':
                value = value[0:max_len]
            elif type == 'pre':
                value = value[len(value) - max_len:len(value)]
        return Sequence(value)

    def cut_sequences(self, words: list[Sequence], max_len=None, type='post'):
        cutted_seqs = list(map(lambda x: self.cut(
            x, type, max_len), words))
        return cutted_seqs
    
    def generate_all_words(self):
        list_symbols = list(self._alphabet.symbols)
        list_symbols.sort()
        ret = [self.data[0]]
        counter = 1
        while len(ret) > 0:
            result = ret.pop(0)
            yield result
            for symbol in list_symbols:
                value = list(result.value)
                value.append(symbol)
                extension = Sequence(value)
                ret.append(extension)
            ret.append(self.data[counter])
            counter = (1 + counter) % len(self.data)