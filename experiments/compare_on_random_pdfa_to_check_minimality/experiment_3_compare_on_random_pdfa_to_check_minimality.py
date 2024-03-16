from pymodelextractor.learners.observation_table_learners.pdfa_lstar_learner import PDFALStarLearner
from pymodelextractor.learners.observation_table_learners.pdfa_lstarcol_learner import PDFALStarColLearner
from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import PDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.model_comparators.wfa_quantization_comparison_strategy import WFAQuantizationComparator
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.symbol import SymbolStr
from pymodelextractor.teachers.pdfa_teacher import PDFATeacher 
from pythautomata.utilities.pdfa_operations import check_is_minimal
from pythautomata.utilities import pdfa_generator
from pythautomata.utilities import nicaud_dfa_generator
#from utilities.utils import compute_stats
from utilities.hypothesis_aware_sample_probabilistic_teacher import HypothesisAwareSampleProbabilisticTeacher
from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from pymodelextractor.teachers.sample_probabilistic_teacher import SampleProbabilisticTeacher
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner, QuantizationProbabilityPartitioner, RankingPartitioner,QuantizationProbabilityPartitionerPlus
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator

from utilities.synchronized_pdfa_teacher import SynchronizedPDFATeacher

import numpy as np
import pandas as pd
import datetime
from  utilities import utils, constants
import joblib
import os
from tqdm import tqdm
from functools import partial

#Experiment to compare WLStar and QuaNT
def generate_and_persist_random_PDFAs():
    path = './experiments/compare_on_random_pdfa_to_check_minimality/instances/'
    try:
        pdfas = utils.load_pdfas(path)
        if len(pdfas) == 0:
            assert(False)
        print('Instances succesfully loaded!')
    except:
        print('Failed loading instances!')
        print('Generating instances...')
        sizes = [50]
        n= 100
        counter = 0
        pdfas = []
        pbar = tqdm(total=n*len(sizes))
        for size in sizes:
            counter = 0
            for i in range(n):
                dfa = nicaud_dfa_generator.generate_dfa(alphabet = constants.binaryAlphabet, nominal_size = size, seed = counter)                
                pdfa = pdfa_generator.pdfa_from_dfa(dfa, zero_probability=0.3)   
                pdfa.name = "random_PDFA_nominal_size_"+str(size)+"_"+str(counter)             
                pdfas.append(pdfa)
                joblib.dump(pdfa, filename = pdfa+dfa.name)
                counter += 1    
                pbar.update(1) 
        pbar.close() 
    return pdfas

def get_masked_pdfa_teacher(pdfa, comparator):
    undefined_ouput = np.zeros(len(pdfa.alphabet)+1)
    synchronic_model = SyncronicModelGuidedLanguageModel(pdfa, guiding_model=None, model_name= pdfa.name+"_SYNCH", max_seq_length=10, 
                                                         normalize_outputs=False, top_k=len(pdfa.alphabet)+1, check_is_defined=True, 
                                                         undefined_ouput=undefined_ouput)
    return SampleProbabilisticTeacher(synchronic_model, comparator = comparator, sample_size = 100, max_seq_length = 25)

def get_masked_pdfa_exact_teacher(pdfa, comparator):
    undefined_ouput = np.zeros(len(pdfa.alphabet)+1)
    synchronic_model = SyncronicModelGuidedLanguageModel(pdfa, guiding_model=None, model_name= pdfa.name+"_SYNCH", max_seq_length=10, 
                                                         normalize_outputs=False, top_k=len(pdfa.alphabet)+1, check_is_defined=True, 
                                                         undefined_ouput=undefined_ouput)
    return SynchronizedPDFATeacher(synchronic_model, pdfa, comparison_strategy = comparator)

def experiment_random_PDFAS():
    print(os.listdir())    
    pdfas = generate_and_persist_random_PDFAs()
    partitions = 10
    max_seq_length = 25
    #max_seconds_run = None
    partitioner = QuantizationProbabilityPartitionerPlus(partitions)
    partition_comparator = WFAPartitionComparator(partitioner)
    partition_comparator_omit_zero = WFAPartitionComparator(partitioner, omit_zero_transitions=True)
    hypothesis_aware_teacher = partial(HypothesisAwareSampleProbabilisticTeacher,  comparator = partition_comparator, sample_size = 100, max_seq_length = max_seq_length)
    standard_sample_teacher = partial(SampleProbabilisticTeacher, comparator = partition_comparator, sample_size = 100, max_seq_length = 25)
    filter_sample_teacher = partial(get_masked_pdfa_teacher, comparator = partition_comparator)
    filter_exact_teacher = partial(get_masked_pdfa_exact_teacher, comparator = partition_comparator_omit_zero)
    pdfa_teacher_standard = partial(PDFATeacher, comparison_strategy = partition_comparator)
    pdfa_teacher_omit_zero = partial(PDFATeacher, comparison_strategy = partition_comparator_omit_zero)
    algorithms = [
        ('QuantNaryTreeLearner_Omit_Zero_Transitions', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = True, probabilityPartitioner = partitioner), hypothesis_aware_teacher),
        ('QuantNaryTreeLearner_Teacher_Filter', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = False, probabilityPartitioner = partitioner), filter_sample_teacher),
        #('QuantNaryTreeLearner_Standard_Teacher', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = False, probabilityPartitioner = partitioner), standard_sample_teacher),
        #SE CAE ESTE -> ('QuantNaryTreeLearner_Omit_Zero_Transitions_AND_Teacher_Filter', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = True, probabilityPartitioner = partitioner), sample_teacher),
        ('QuantNaryTreeLearner_Omit_Zero_Transitions_exact_teacher', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = True, probabilityPartitioner = partitioner), pdfa_teacher_omit_zero),
        ('QuantNaryTreeLearner_Teacher_Filter_exact', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = False, probabilityPartitioner = partitioner), filter_exact_teacher),
        #('QuantNaryTreeLearner_Standard_Teacher_exact_teacher', partial(PDFAQuantizationNAryTreeLearner, omit_zero_transitions = False, probabilityPartitioner = partitioner), pdfa_teacher_standard)
                     ]
        
    results = []   
    number_of_executions  = 11
    
    print('Excecuting extraction...')
    pbar = tqdm(total=number_of_executions*len(algorithms)*len(pdfas))
    for (algorithm_name,algorithm, teacher) in algorithms:
        for pdfa in pdfas:
            
            sg = UniformLengthSequenceGenerator(pdfa.alphabet, max_seq_length, random_seed=42)
            sequences_anywhere = sg.generate_words(1000)              
            sg2 = GuidingWDFASequenceGenerator(pdfa, max_seq_length, random_seed=42)
            sequences_in_target = sg2.generate_words(1000)   
            for i in range(number_of_executions):
                pdfa_teacher = teacher(pdfa)
                learner = algorithm(check_probabilistic_hipothesis = False)
                secs, result = utils.time_fun(learner.learn,pdfa_teacher)               
                pbar.update(1)                     
                if i > 0:
                    if result.info['observation_tree'] is None:
                        tree_depth = 0
                        inner_nodes = 0
                    else:
                        tree_depth = result.info['observation_tree'].depth
                        inner_nodes = len(result.info['observation_tree'].inner_nodes)
                    extracted_model = result.model                    
                    accuracy_anywhere = utils.partial_accuracy(target_model=pdfa, partial_model=extracted_model, partitioner = learner.probability_partitioner, test_sequences=sequences_anywhere)['Accuracy']
                    accuracy_in_target = utils.partial_accuracy(target_model=pdfa, partial_model=extracted_model, partitioner = learner.probability_partitioner, test_sequences=sequences_in_target)['Accuracy']
                    partition_comparator = WFAPartitionComparator(learner.probability_partitioner)
                    partition_comparator_omit_zero = WFAPartitionComparator(learner.probability_partitioner, omit_zero_transitions=True)
                    is_minimal = check_is_minimal(extracted_model)
                    is_equivalent_exact = partition_comparator.are_equivalent(pdfa, extracted_model)
                    is_equivalent_omit_zero = partition_comparator_omit_zero.are_equivalent(pdfa, extracted_model)
                    results.append((algorithm_name, pdfa.name, len(pdfa.weighted_states), len(extracted_model.weighted_states), i, secs, result.info['last_token_weight_queries_count'], result.info['equivalence_queries_count'], tree_depth, inner_nodes, accuracy_in_target, accuracy_anywhere, is_equivalent_exact, is_equivalent_omit_zero, is_minimal))
    pbar.close() 
    dfresults = pd.DataFrame(results, columns = ['Algorithm', 'Instance', 'Number of States', 'Extracted Number of States','RunNumber','Time(s)','LastTokenQuery', 'EquivalenceQuery', 'Tree Depth', 'Inner Nodes','Accuracy_in_target','Accuracy_anywhere', 'IsEquivalentExact', 'IsEquivalentOmitZero', 'IsMinimal']) 
    dfresults.to_csv('./experiments/compare_on_random_pdfa_to_check_minimality/results/results_'+datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+'.csv') 

def run():
    experiment_random_PDFAS()
