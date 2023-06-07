from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr

binaryAlphabet = Alphabet(frozenset((SymbolStr('0'), SymbolStr('1'))))
zero = binaryAlphabet['0']
one = binaryAlphabet['1']   

def get_n_ary_alphabet(n):
    alphabet = []
    for i in range(n):
        alphabet.append(SymbolStr(str(i)))
    return Alphabet(frozenset(alphabet))