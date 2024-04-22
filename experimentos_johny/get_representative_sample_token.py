from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton as PDFA
import numpy as np

from pythautomata.utilities.moore_machine_minimizer import MooreMachineMinimizer
from pythautomata.automata.moore_machine_automaton import MooreMachineAutomaton
from pythautomata.base_types.moore_state import MooreState
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet

def get_representative_sample_token(pdfa: PDFA, sample_size: int, max_tokens: int, retry: False):
    assert (sample_size >= 0)
    sample = list()
    for i in range(sample_size):
        sample.append(generate_single_word(pdfa, max_tokens, retry))
    return sample


def generate_single_word_up_to_max_tokens(pdfa: PDFA, max_tokens: int):
    tokens_count = 0
    word = Sequence()
    first_state = list(filter(lambda x: x.initial_weight ==
                       1, pdfa.weighted_states))[0]
    symbols, weights, next_states = first_state.get_all_symbol_weights()
    next_symbol = np.random.choice(symbols, p=weights)
    tokens_count +=1
    while next_symbol != pdfa.terminal_symbol and tokens_count <= max_tokens:
        word += next_symbol
        i = symbols.index(next_symbol)
        next_state = next_states[i]
        symbols, weights, next_states = next_state.get_all_symbol_weights()
        next_symbol = np.random.choice(symbols, p=weights)          
        is_valid = next_symbol != pdfa.terminal_symbol
        tokens_count +=1
    return word, is_valid

def generate_single_word(pdfa, max_tokens, retry):
    word, is_valid =  generate_single_word_up_to_max_tokens(pdfa, max_tokens)
    while not is_valid and retry:
        word, is_valid =  generate_single_word_up_to_max_tokens(pdfa, max_tokens)
    return word
