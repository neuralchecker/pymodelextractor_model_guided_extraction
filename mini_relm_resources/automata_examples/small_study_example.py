from pythautomata.automata.deterministic_finite_automaton import \
    DeterministicFiniteAutomaton
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.state import State
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.model_comparators.dfa_comparison_strategy import \
    DFAComparisonStrategy as DFAComparator
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton


alphabet = Alphabet(frozenset((SymbolStr("I studied"), SymbolStr("medicine"), SymbolStr("astrophysics"))))

# This automaton represents the following regex: "I studied (medicine|astrophysics)"
def get_small_study_example(terminal_symbol):
    stateA = WeightedState("A", 1,0, terminal_symbol)
    stateB = WeightedState("B", 0,0, terminal_symbol)
    stateA.add_transition(SymbolStr("I studied"), stateB, 1)
    stateC = WeightedState("C", 0,1, terminal_symbol)
    stateB.add_transition(SymbolStr("medicine"), stateC, 1)
    stateB.add_transition(SymbolStr("astrophysics"), stateC, 1)

    hole = WeightedState("hole", 0, 0, terminal_symbol)
    
    states = frozenset({stateA, stateB, stateC, hole})

    for state in states:
        for symbol in alphabet.symbols:
            if symbol not in state.transitions_set:
                state.add_transition(symbol, hole, 0)


    comparator = None
    return ProbabilisticDeterministicFiniteAutomaton(alphabet, states, terminal_symbol, comparator, "Small_Study_Example", check_is_probabilistic = False)    