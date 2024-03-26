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


alphabet = Alphabet(frozenset((SymbolStr("The"), SymbolStr("man"), SymbolStr("woman"), SymbolStr("was trained in"), 
                               SymbolStr("medicine"), SymbolStr("science"), SymbolStr("engineering"), 
                               SymbolStr("maths"), SymbolStr("art"), SymbolStr("music"), SymbolStr("astrophysics"),
                               SymbolStr("astrology"))))

# This automaton represents the following regex: "The (man|woman) was trained in (medicine|science|engineering|maths|art|music)"
def get_man_woman_wfa(terminal_symbol):
    stateA = WeightedState("A", 1,0, terminal_symbol)
    stateB = WeightedState("B", 0,0, terminal_symbol)
    stateA.add_transition(SymbolStr("The"), stateB, 1)
    stateC = WeightedState("C", 0,0, terminal_symbol)
    stateB.add_transition(SymbolStr("man"), stateC , 1)
    stateB.add_transition(SymbolStr("woman"), stateC, 1)
    stateD = WeightedState("D", 0,0, terminal_symbol)
    stateC.add_transition(SymbolStr("was trained in"), stateD, 1)
    stateE = WeightedState("E", 0,1, terminal_symbol)
    stateD.add_transition(SymbolStr("medicine"), stateE, 1)
    stateD.add_transition(SymbolStr("science"), stateE, 1)
    stateD.add_transition(SymbolStr("engineering"), stateE, 1)
    stateD.add_transition(SymbolStr("maths"), stateE, 1)
    stateD.add_transition(SymbolStr("art"), stateE, 1)
    stateD.add_transition(SymbolStr("music"), stateE, 1)
    stateD.add_transition(SymbolStr("astrophysics"), stateE, 1)
    stateD.add_transition(SymbolStr("astrology"), stateE, 1)

    hole = WeightedState("hole", 0, 0, terminal_symbol)
    
    states = frozenset({stateA, stateB, stateC, stateD, stateE, hole})

    for state in states:
        _, weights, _ = state.get_all_symbol_weights()
        total_weights = sum(weights)
        for symbol in alphabet.symbols:
            if symbol not in state.transitions_set:
                state.add_transition(symbol, hole, 0)


    comparator = None
    return ProbabilisticDeterministicFiniteAutomaton(alphabet, states, terminal_symbol, comparator, "Man_Woman_WFA", check_is_probabilistic = False)    