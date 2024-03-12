from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner
from pythautomata.model_exporters.partition_mapper import TopKPartitionMapper
from pythautomata.model_exporters.image_exporters.wfa_image_exporter_with_partition_mapper import WFAImageExporterWithPartitionMapper
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.automata_definitions.weighted_tomitas_grammars import WeightedTomitasGrammars

from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher

from utilities.wfa_dot_exporter_with_partition_mapper import WFADotExporterWithPartitionMapper
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner

from experiments.tata_box import tata_box_extraction_experiment
from experiments.tata_box_model_guided_extraction import tata_box_model_guided_extraction
from experiments.hdfs import hdfs_extraction_experiment
from experiments.hdfs_by_time import hdfs_by_time_extraction_experiment
from experiments.good_bad_queries_syncronic_excecution import good_bad_queries_syncronic_excecution_experiment
from experiments.compare_on_random_pdfa import experiment_1_compare_on_random_pdfa
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

import sys


if __name__ == '__main__':
    #args = sys.argv   
    #run_type = int(args[1])  
    run_type =1
    if run_type == 1:
        experiment_1_compare_on_random_pdfa.run()
    # elif run_type == 2:
    #     hdfs_extraction_experiment.run_test()
    # elif run_type == 3:
    #     hdfs_by_time_extraction_experiment.run_test()
    # elif run_type == 4:
    #     good_bad_queries_syncronic_excecution_experiment.run_test()
    