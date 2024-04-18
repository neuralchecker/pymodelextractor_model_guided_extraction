from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton as PDFA
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from typing import Union

from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel 

class SynchronizedPDFATeacher(ProbabilisticTeacher):

    def __init__(self, synchronic_model: SyncronicModelGuidedLanguageModel, pdfa: WeightedAutomaton, comparison_strategy: FiniteAutomataComparator):
        super().__init__(synchronic_model)
        self.pdfa =pdfa
        self._comparison_strategy = comparison_strategy
        #self.__target_pdfa_model = model

    def sequence_weight(self, sequence: Sequence):
        raise NotImplemented
        return self.__target_pdfa_model.sequence_weight(sequence)

    def log_sequence_weight(self, sequence: Sequence):
        raise NotImplemented
        return self.__target_pdfa_model.log_sequence_weight(sequence)

    def get_log_probability_error(self, seq, aut: PDFA):
        raise NotImplemented
        return abs(aut.log_sequence_weight(seq) - self.log_sequence_weight(seq))

    def equivalence_query(self, aut: PDFA) -> tuple[bool, Union[Sequence, None]]:
        self._equivalence_queries_count += 1
        counterexample = self._comparison_strategy.get_counterexample_between(aut, self.pdfa)
        are_equivalent = counterexample is None
        return are_equivalent, counterexample

    @property
    def alphabet(self):
        return self.pdfa.alphabet

    @property
    def terminal_symbol(self):
        return self.pdfa.terminal_symbol