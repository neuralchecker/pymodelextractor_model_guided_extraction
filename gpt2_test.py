import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
#import sys
#sys.path.append("../")
#sys.path.append("../../")
import os

# get the current working directory
current_working_directory = os.getcwd()

# print output to the console
print(current_working_directory)

from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner, QuantizationProbabilityPartitioner, RankingPartitioner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator

torch.manual_seed(42)




from case_studies.gpt2.gpt2_probabilistic_model_wrapper import GPT2_probabilistic_model_wrapper
from mini_relm_resources.automata_examples.man_woman_wfa import alphabet

from mini_relm_resources.automata_examples.man_woman_wfa import get_man_woman_wfa

from pythautomata.utilities.pdfa_operations import get_representative_sample

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.utilities.pdfa_operations import get_representative_sample
import sys
sys.path.append("../")

class PDFASequenceGenerator(SequenceGenerator):    
    def __init__(self, pdfa, max_seq_length: int, random_seed: int = 21):
        self.pdfa = pdfa
        super().__init__(pdfa.alphabet, max_seq_length, random_seed)
    
    def generate_words(self, number_of_words: int):
        return get_representative_sample(self.pdfa, number_of_words)

    def generate_single_word(self, length):
        raise NotImplementedError

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id).to(device)

    wrapper = GPT2_probabilistic_model_wrapper(50, alphabet, device, model, tokenizer)


    property_model = get_man_woman_wfa(wrapper.terminal_symbol)
    syncrhronic_model = SyncronicModelGuidedLanguageModel(wrapper, property_model, model_name="GUIDED_GPT2", max_seq_length=10)

    partitioner = TopKProbabilityPartitioner(1)
    comparator = WFAPartitionComparator(partitioner)
    epsilon = 0.1
    delta = epsilon
    sequence_generator = PDFASequenceGenerator(property_model, max_seq_length=10)
    max_states = 10
    max_query_length = 10



    teacher  = PACProbabilisticTeacher(syncrhronic_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=sequence_generator, compute_epsilon_star=False)
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, None, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True,  check_probabilistic_hipothesis = False)


    learning_result = learner.learn(teacher, verbose=True)




