from pythautomata.base_types.symbol import Symbol, SymbolStr
from pythautomata.base_types.alphabet import Alphabet


class SymbolEncoder():

    def __init__(self, alphabet: Alphabet, terminal_symbol = None, padding_symbol = None):
        self.alphabet = alphabet
        symbols = list(map(lambda x: x.value, alphabet.symbols))
        symbols.sort(reverse=False)
        
        if terminal_symbol is not None: 
            assert(terminal_symbol not in symbols)
            symbols = symbols + [terminal_symbol.value]
        if padding_symbol is not None: 
            assert(padding_symbol not in symbols)
            symbols = [padding_symbol.value] + symbols

        ints = range(0, len(symbols))
        self.encoding_dict = dict(zip(symbols, ints))
        self.decoding_dict = dict(zip(ints, symbols))

    def encode(self, symbol):
        return self.encoding_dict[symbol.value]

    def encode_sequence(self, sequence):
        return list(map(lambda x: self.encode(x), sequence))

    def decode(self, number):
        return self.decoding_dict[number]
