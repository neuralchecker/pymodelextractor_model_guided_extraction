from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator

from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner, QuantizationProbabilityPartitioner, RankingPartitioner
from utilities.relaxed_quantization_probability_partitioner import RelaxedQuantizationProbabilityPartitioner

from pythautomata.base_types.symbol import SymbolStr
from pythautomata.model_exporters.dot_exporters.wfa_dot_exporting_strategy import WFADotExportingStrategy 

from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pymodelextractor.teachers.pac_batch_probabilistic_teacher import PACBatchProbabilisticTeacher

from case_studies.good_bad_queries.good_bad_quereies_language_model import GoodBadQueriesLanguageModel
from tqdm import tqdm
from utilities import utils
from data.data_loaders.good_bad_queries_data_loader import GoodBadQueriesDataLoader
from utilities.data_sequence_generator import DataSequenceGenerator
from utilities.good_bad_queries_has_attack_symbol_partitioner import GoodBadQueriesHasAttackSymbolProbabilityPartitioner
from utilities.syncronic_excecution_language_model import SyncronicExcecutionLanguageModel
from utilities.wfa_dot_exporter_with_partition_mapper import WFADotExporterWithPartitionMapper
from utilities.syncronic_topk_probability_mapper import SyncronicTopKProbabilityMapper
from utilities.syncronic_quantization_probability_mapper import SyncronicQuantizationProbabilityMapper

import numpy as np
import pandas as pd
import datetime
import random

def get_models():
    model1 = GoodBadQueriesLanguageModel("model.04-03-2022", 10)
    model2 = GoodBadQueriesLanguageModel("model.26-07-2021", 10)
    partitioner = RelaxedQuantizationProbabilityPartitioner(1)
    return [SyncronicExcecutionLanguageModel(model1, model2, partitioner, model_name = "B"), 
            SyncronicExcecutionLanguageModel(model1, model2, partitioner, model_name = "T", compose_by_difference = True)]

def get_algorithms(model):
    partitioner = RelaxedQuantizationProbabilityPartitioner(1)
    comparator = WFAPartitionComparator(partitioner)

    learner = BoundedPDFAQuantizationNAryTreeLearner
    return [('BoundedQuant_top1',learner, partitioner, comparator)]

def get_test_data(n_samples):
    data = GoodBadQueriesDataLoader('./data/good_bad_queries',10).data
    
    some_good_data = random.sample(data['good'], int(n_samples/2))
    some_bad_data = random.sample(data['bad'], int(n_samples/2))
    result = []
    result.extend(some_good_data)
    result.extend(some_bad_data)
    return result  

def run_test():
    results_list = []   
    number_of_executions  = 1
    models_to_test = get_models()
    algorithms = get_algorithms(models_to_test[0])
    data_loader = GoodBadQueriesDataLoader('./data/good_bad_queries',10)
    data = data_loader.data['bad']
    sequence_generator = DataSequenceGenerator(data = data, alphabet=data_loader.alphabet, max_seq_length=10)
    test_data = sequence_generator.generate_words(1000)
    epsilon = 0.05
    delta = 0.05
    max_states = 500000
    max_query_length = 100000
    max_seconds_run=[120]

    exp_results_parent_path='./experiments/good_bad_queries_syncronic_excecution/results'
    learned_model_path = './experiments/good_bad_queries_syncronic_excecution/results/learned_models'
    results_path = utils.get_path_for_result_file(exp_results_parent_path)

    print('Excecuting extraction...')
    pbar = tqdm(total=number_of_executions*len(algorithms)*len(models_to_test)*len(max_seconds_run))
    for (algorithm_name,algorithm, partitioner, comparator) in algorithms:
        for model in models_to_test:
            for max_secs in max_seconds_run:
                for i in range(number_of_executions):
                    teacher  = PACBatchProbabilisticTeacher(model, epsilon = epsilon, delta = delta, max_seq_length = model._max_seq_length, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False)
                    learner = algorithm(partitioner, max_states, max_query_length, max_secs, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True, check_probabilistic_hipothesis = False)
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
                        extracted_model.name = algorithm_name+"_"+model.name+"_"+str(i)+"_"+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                        WFADotExportingStrategy().export(extracted_model, learned_model_path)
                        WFADotExporterWithPartitionMapper(partitioner,SyncronicQuantizationProbabilityMapper(),pretty=True).export(extracted_model, learned_model_path, file_name=extracted_model.name+"_mapped")
                        
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
 
