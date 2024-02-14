import numpy as np
from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner
from mini_relm_resources.automata_examples import floating_point_wfa 

class FloatingPointProbabilityPartitioner(ProbabilityPartitioner):
    def __init__(self) -> None:     
        super().__init__()

    def _get_partition(self, probability_vector):
        dot_proba = probability_vector[floating_point_wfa._get_symbol_index(floating_point_wfa.dot)]
        if dot_proba>0:
            return [0]
        max_digit_proba = 0
        for digit in floating_point_wfa.numbers:
            digit_proba = probability_vector[floating_point_wfa._get_symbol_index(digit)]
            max_digit_proba = max(digit_proba, max_digit_proba)
        if max_digit_proba>0:
            return [1]
        return [2]