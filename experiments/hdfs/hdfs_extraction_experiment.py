from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner, QuantizationProbabilityPartitioner, RankingPartitioner
from utilities.tata_box_partitioner import TataBoxProbabilityPartitioner
#from pythautomata.model_exporters.wfa_image_exporter_with_partition_mapper import WFAImageExporterWithPartitionMapper
from pythautomata.base_types.symbol import SymbolStr

from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher

from case_studies.good_bad_queries.good_bad_quereies_language_model import GoodBadQueriesLanguageModel
from tqdm import tqdm
from utilities import utils
from data.data_loaders.deeplog_data_loader import DeepLogDataLoader
from utilities.data_sequence_generator import DataSequenceGenerator
from utilities.neural_networks.rnn_language_models.last_token_language_model import LastTokenLanguageModel

import numpy as np
import pandas as pd
import datetime
import random

def get_models():
    path="neural_networks/trained_models/hdfs_last_token_language_model/"
    return [LastTokenLanguageModel(path)]

def get_algorithms(model):
    partitioner1 = TopKProbabilityPartitioner(6)
    comparator1 = WFAPartitionComparator(partitioner1)

    partitioner2 = RankingPartitioner(6)
    comparator2 = WFAPartitionComparator(partitioner2)

    
    partitioner3 = QuantizationProbabilityPartitioner(10)
    comparator3 = WFAPartitionComparator(partitioner3)

    partitioner4 = QuantizationProbabilityPartitioner(5)    
    comparator4 = WFAPartitionComparator(partitioner4)

    partitioner5 = RankingPartitioner(3)    
    comparator5 = WFAPartitionComparator(partitioner5)

    partitioner6 = TopKProbabilityPartitioner(3)
    comparator6 = WFAPartitionComparator(partitioner6)

    partitioner7 = QuantizationProbabilityPartitioner(3)
    comparator7 = WFAPartitionComparator(partitioner7)

    learner = BoundedPDFAQuantizationNAryTreeLearner
    
    return [('BoundedQuant_top6',learner, partitioner1, comparator1),
            ('BoundedQuant_rank6',learner, partitioner2, comparator2),
            ('BoundedQuant_quant10',learner, partitioner3, comparator3),
            ('BoundedQuant_rank3',learner, partitioner4, comparator4),
            ('BoundedQuant_quant5',learner, partitioner5, comparator5),
            ('BoundedQuant_top3',learner, partitioner6, comparator6),
             ('BoundedQuant_quant3',learner, partitioner7, comparator7)]

def get_test_data(n_samples):
    data = DeepLogDataLoader("./data/deep_log/hdfs_train").data    
    result = random.sample(data, n_samples)    
    return result  

def run_test():
    results_list = []   
    number_of_executions  = 1
    models_to_test = get_models()
    algorithms = get_algorithms(models_to_test[0])
    data_loader = DeepLogDataLoader("./data/deep_log/hdfs_train")
    data = data_loader.data
    sequence_generator = DataSequenceGenerator(data = data, alphabet=data_loader.alphabet, max_seq_length=20)
    test_data = sequence_generator.generate_words(1000)
    epsilon = 0.05
    delta = 0.05
    max_states = 10000
    max_query_length = 1000
    max_seconds_run=[10000]
    #max_seconds_run = [None]

    exp_results_parent_path='./experiments/hdfs/results'
    learned_model_path = './experiments/hdfs/results/learned_models'
    results_path = utils.get_path_for_result_file(exp_results_parent_path)

    print('Excecuting extraction...')
    pbar = tqdm(total=number_of_executions*len(algorithms)*len(models_to_test)*len(max_seconds_run))
    for (algorithm_name,algorithm, partitioner, comparator) in algorithms:
        for model in models_to_test:
            for max_secs in max_seconds_run:
                for i in range(number_of_executions):
                    teacher  = PACBatchProbabilisticTeacher(model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False)
                    learner = algorithm(partitioner, max_states, max_query_length, max_secs, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True,  check_probabilistic_hipothesis = False)
                    secs, learning_result = utils.time_fun(learner.learn,teacher)     
                    pbar.update(1)                        
                    if i >= 0:
                        if learning_result.info['observation_tree'] is None:
                            tree_depth = 0
                            inner_nodes = 0
                        else:
                            tree_depth = learning_result.info['observation_tree'].depth
                            inner_nodes = len(learning_result.info['observation_tree'].inner_nodes)
                        extracted_model = learning_result.model
                        extracted_model.name = algorithm_name+"_"+str(i)+"_"+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                        #extracted_model.export(learned_model_path)
                        result = dict()
                        result.update({'Algorithm':algorithm_name, 
                                    'Instance': model.name,
                                        'Number of Extracted States': len(extracted_model.weighted_states) ,
                                        'RunNumber': i,
                                        'Time (s)': secs,
                                        'LastTokenQuery': learning_result.info['last_token_weight_queries_count'], 
                                        'EquivalenceQuery': learning_result.info['equivalence_queries_count'], 
                                        'NumberOfStatesExceeded': learning_result.info['NumberOfStatesExceeded'],
                                        'QueryLengthExceeded':learning_result.info['QueryLengthExceeded'], 
                                        'TimeExceeded': learning_result.info['TimeExceeded'],
                                        'Tree Depth': tree_depth,
                                        'Inner Nodes': inner_nodes,
                                        'TimeBound': max_secs
                                    })
                        metrics = utils.compute_stats(model, extracted_model, partitioner, test_sequences=test_data)
                        result.update(metrics)
                        results_list.append(result)
                        dfresults = pd.DataFrame(results_list, columns = results_list[0].keys()) 
                        dfresults.to_csv(exp_results_parent_path+'/results_backup.csv') 
                pbar.close() 
                dfresults = pd.DataFrame(results_list, columns = results_list[0].keys()) 
                dfresults.to_csv(results_path)
 
