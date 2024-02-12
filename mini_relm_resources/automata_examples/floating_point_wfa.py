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



dot = SymbolStr(".")
zero = SymbolStr("0")
one = SymbolStr("1")
two = SymbolStr("2")
three = SymbolStr("3")
four = SymbolStr("4")
five = SymbolStr("5")
six = SymbolStr("6")
seven = SymbolStr("7")
eight = SymbolStr("8")
nine = SymbolStr("9")

numbers = [zero, one, two, three, four, five, six, seven, eight, nine]
symbols = numbers.copy()
symbols.append(dot)
alphabet = Alphabet(frozenset(symbols))
# This automaton represents the following regex: "The (man|woman) studied (medicine|science|engineering|maths|art|music)"
def get_floating_point_wfa(terminal_symbol):
    stateA = WeightedState("initial", 1,0, terminal_symbol)
    stateB = WeightedState("numbers", 0,1, terminal_symbol)
    for number in numbers:
        stateA.add_transition(number, stateB, 1)
    for number in numbers:
        stateB.add_transition(number, stateB, 1)
    stateC = WeightedState("dot", 0,0, terminal_symbol)
    stateA.add_transition(dot, stateC, 1)
    stateB.add_transition(dot, stateC, 1)
    stateD = WeightedState("more_numbers", 0,1, terminal_symbol)
    for number in numbers:
        stateC.add_transition(number, stateD, 1)
    for number in numbers:
        stateD.add_transition(number, stateD, 1)
    
    hole = WeightedState("hole", 0, 0, terminal_symbol)
    
    states = frozenset({stateA, stateB, stateC, stateD, hole})

    for state in states:
        for symbol in alphabet.symbols:
            if symbol not in state.transitions_set:
                state.add_transition(symbol, hole, 0)


    comparator = None
    return ProbabilisticDeterministicFiniteAutomaton(alphabet, states, terminal_symbol, comparator, "Man_Woman_WFA", check_is_probabilistic = False)    
