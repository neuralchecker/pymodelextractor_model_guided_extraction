import torch
import time
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from case_studies.gpt2.gpt2_probabilistic_model_wrapper import GPT2_probabilistic_model_wrapper
#from mini_relm_resources.automata_examples.floating_point_wfa_01_all_tokens import alphabet
from mini_relm_resources.automata_examples.floating_point_wfa_01 import alphabet

from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
#from mini_relm_resources.automata_examples.floating_point_wfa_01_all_tokens import get_floating_point_wfa_01_all_tokens
from mini_relm_resources.automata_examples.floating_point_wfa_01 import get_floating_point_wfa_01
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from utilities.hypothesis_aware_sample_probabilistic_teacher import HypothesisAwareSampleProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator
from pythautomata.utilities.guiding_pdfa_sequence_generator import GuidingPDFASequenceGenerator
from pythautomata.utilities.pdfa_operations import get_representative_sample
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator
from sampling.get_representative_sample_length import get_representative_sample_length
from sampling.get_representative_sample_token import get_representative_sample_token

from pythautomata.utilities.probability_partitioner import QuantizationProbabilityPartitionerPlus
from pythautomata.model_exporters.dot_exporters.wfa_dot_exporting_strategy import WFADotExportingStrategy
from utilities.floating_point_partitioner import FloatingPointProbabilityPartitioner

from pythautomata.model_exporters.image_exporters.image_exporting_strategy import ImageExportingStrategy
import joblib 


def sample_floating_point():
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    wrapper = GPT2_probabilistic_model_wrapper(50, alphabet, device, model, tokenizer)
    property_model = get_floating_point_wfa_01(wrapper.terminal_symbol)
    #property_model = get_floating_point_wfa_01_all_tokens(wrapper.terminal_symbol)
    synchronic_model = SyncronicModelGuidedLanguageModel(wrapper, property_model, model_name="GUIDED_GPT2", max_seq_length=10, normalize_outputs=True)
    guiding_generator = GuidingWDFASequenceGenerator(property_model, None)
    #guiding_generator = UniformWordSequenceGenerator(alphabet, 6)
    #partitioner = FloatingPointProbabilityPartitioner()
    partitioner = QuantizationProbabilityPartitionerPlus(100)
    comparator = WFAPartitionComparator(partitioner)
    epsilon = 0.05
    delta = epsilon
    sequence_generator = guiding_generator
    max_states = 15
    max_query_length = 6
    teacher = HypothesisAwareSampleProbabilisticTeacher(synchronic_model, comparator = comparator, max_seq_length = 4, sample_size = 100)
    #teacher  = PACProbabilisticTeacher(syncrhronic_model, epsilon = epsilon, delta = delta, max_seq_length = 100, comparator = comparator, sequence_generator=guiding_generator, compute_epsilon_star=False)
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner = partitioner, max_states = max_states, max_query_length = max_query_length, max_seconds_run = 30, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True,  check_probabilistic_hipothesis = True, mean_distribution_for_partial_hipothesis = True, omit_zero_transitions = True)

    
    exporter = WFADotExportingStrategy()
    #exporter.export(property_model)

    learning_result = learner.learn(teacher, verbose=True)
    
    pdfa = learning_result.model
    joblib.dump(pdfa, "pdfa")
    exporter.export(pdfa)
    #ImageExportingStrategy(exporter, "svg").export(pdfa, "./")
    floating_points = []
  
    sample_time = 0
    start_time = time.time()
    for i in range(10000):
        #number = get_representative_sample_length(pdfa, sample_size = 1, length=1, retry=False)
        #number = get_representative_sample_token(pdfa, sample_size = 1, max_tokens = 2, retry=True)
        number = get_representative_sample(pdfa, sample_size = 1)
        number_string = str(number)
        
        result = number_string.replace('[', '').replace(']', '').replace(',', '')

        floating_points.append(result)

    sample_time = time.time() - start_time

    df = pd.DataFrame(floating_points, columns=["floating-point"])
    df.to_csv("floating_points_asmr.csv", index=False)



    
def get_gpt2_model_and_tokenizer():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_prefix_space=False)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id).to(device)
                                                
    return model_id, model, tokenizer, device



if __name__ == "__main__":
    sample_floating_point()