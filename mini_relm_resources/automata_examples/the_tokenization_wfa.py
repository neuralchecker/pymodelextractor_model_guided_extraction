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


alphabet = Alphabet(frozenset((SymbolStr("The"), SymbolStr("T"), SymbolStr("h"), SymbolStr("e"), 
                               SymbolStr("he"),  SymbolStr("Th"),SymbolStr("house"), SymbolStr("gorilla"))))

# This automaton represents the following regex: "The (man|woman) studied (medicine|science|engineering|maths|art|music)"
def get_the_tokenization_wfa(terminal_symbol):
    stateStart = WeightedState("start", 1,0, terminal_symbol)
    stateThe = WeightedState("The", 0,0, terminal_symbol)
    stateTh = WeightedState("Th", 0,0, terminal_symbol)
    stateT = WeightedState("T", 0,0, terminal_symbol)
    stateh = WeightedState("h", 0,0, terminal_symbol)
    stateStart.add_transition(SymbolStr("The"), stateThe, 1)
    stateStart.add_transition(SymbolStr("Th"), stateTh, 1)
    stateStart.add_transition(SymbolStr("T"), stateT, 1)
    stateTh.add_transition(SymbolStr("e"), stateThe, 1)
    stateT.add_transition(SymbolStr("he"), stateThe, 1)
    stateT.add_transition(SymbolStr("h"), stateh, 1)
    stateh.add_transition(SymbolStr("e"), stateThe, 1)
    stateF = WeightedState("E", 0,1, terminal_symbol)
    stateThe.add_transition(SymbolStr("house"), stateF, 1)
    stateThe.add_transition(SymbolStr("gorilla"), stateF, 1)

    hole = WeightedState("hole", 0, 0, terminal_symbol)
    
    states = frozenset({stateStart, stateThe, stateTh, stateT, stateF, hole, stateh})

    for state in states:
        _, weights, _ = state.get_all_symbol_weights()
        total_weights = sum(weights)
        for symbol in alphabet.symbols:
            if symbol not in state.transitions_set:
                state.add_transition(symbol, hole, 0)


    comparator = None
    return ProbabilisticDeterministicFiniteAutomaton(alphabet, states, terminal_symbol, comparator, "Man_Woman_WFA", check_is_probabilistic = False)    
