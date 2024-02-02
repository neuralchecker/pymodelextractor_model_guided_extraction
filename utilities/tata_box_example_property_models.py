from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton

def getTataBoxExampleProperty1():
    terminal_symbol = SymbolStr('$')
    symbolA = SymbolStr('A')
    symbolT = SymbolStr('T')
    symbolG = SymbolStr('G')
    symbolC = SymbolStr('C')

    alphabet = Alphabet([symbolA, symbolC, symbolG, symbolT])

    q0 = WeightedState("q0", 1,0)
    q1 = WeightedState("q1", 0, 0)
    q2 = WeightedState("q2", 0, 0)
    q3 = WeightedState("q3", 0, 0)
    q4 = WeightedState("q4", 0, 0)
    hole = WeightedState("hole", 0, 0)

    q0.add_transition(symbolT, q1, 1)
    q1.add_transition(symbolA, q2, 1)
    q1.add_transition(symbolG, q2, 1)
    q2.add_transition(symbolT, q3, 1)
    q3.add_transition(symbolA, q4, 1)
    q3.add_transition(symbolG, q4, 1)
    
    states = {q0, q1, q2, q3, q4, hole}

    for state in states:
        for symbol in alphabet.symbols:
            if symbol not in state.transitions_set:
                state.add_transition(symbol, hole, 0)

    comparator = None
    return ProbabilisticDeterministicFiniteAutomaton(alphabet, states, terminal_symbol, comparator, "EXAMPLE_PROPERTY", check_is_probabilistic = False)

