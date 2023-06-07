from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner
import numpy as np

class RelaxedQuantizationProbabilityPartitioner(ProbabilityPartitioner):

    def __init__(self, number_of_partitions) -> None:
        super().__init__()
        self._partitions = number_of_partitions

    def _get_interval(self, value):
        if value < 0 or value > 1:
            return value
        limits = np.linspace(0, 1, self._partitions+1)
        if value == 1:
            return self._partitions-1
        positions = list(range(len(limits)-1))
        mid_element = int(len(limits)/2)
        while len(positions) > 1:
            if value >= limits[mid_element]:
                positions = positions[int(len(positions)/2):]
            else:
                positions = positions[:int(len(positions)/2)]
            mid_element = positions[int(len(positions)/2)]
        assert (len(positions) == 1)
        return positions[0]

    def _get_partition(self, probability_vector):
        return np.fromiter((self._get_interval(xi) for xi in probability_vector), dtype=int)