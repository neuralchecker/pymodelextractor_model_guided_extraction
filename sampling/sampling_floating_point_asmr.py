import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from case_studies.gpt2.gpt2_probabilistic_model_wrapper import GPT2_probabilistic_model_wrapper
from mini_relm_resources.automata_examples.floating_point_wfa import alphabet
from utilities.floating_point_partitioner import FloatingPointProbabilityPartitioner
from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from mini_relm_resources.automata_examples.floating_point_wfa import get_floating_point_wfa
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator
from pythautomata.utilities.pdfa_operations import get_representative_sample
from utilities.floating_point_partitioner import FloatingPointProbabilityPartitioner
from pythautomata.utilities.uniform_word_sequence_generator import UniformWordSequenceGenerator




def sample_floating_point():
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    wrapper = GPT2_probabilistic_model_wrapper(50, alphabet, device, model, tokenizer)
    property_model = get_floating_point_wfa(wrapper.terminal_symbol)
    syncrhronic_model = SyncronicModelGuidedLanguageModel(wrapper, property_model, model_name="GUIDED_GPT2", max_seq_length=10, normalize_outputs=True)
    guiding_generator = GuidingWDFASequenceGenerator(property_model, None)
    print(guiding_generator.generate_words(100))
    guiding_generator = UniformWordSequenceGenerator(alphabet, 6)
    print(guiding_generator.generate_words(100))
    partitioner = FloatingPointProbabilityPartitioner()
    comparator = WFAPartitionComparator(partitioner)
    epsilon = 0.05
    delta = epsilon
    sequence_generator = guiding_generator
    max_states = 30
    max_query_length = 100
    teacher  = PACProbabilisticTeacher(syncrhronic_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=guiding_generator, compute_epsilon_star=False)
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, None, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True,  check_probabilistic_hipothesis = False)


    learning_result = learner.learn(teacher, verbose=True)
    pdfa = learning_result.model
    
    floating_points = []
  
    for i in range(10000):
        number = get_representative_sample(pdfa, 1)
        number_string = str(number)
        
        result = number_string.replace('[', '').replace(']', '').replace(',', '')

        floating_points.append(result)
        print(f"i: {i}, Floating Point: {floating_points[-1]}")


    df = pd.DataFrame(floating_points, columns=["floating-point"])
    df.to_csv("floating_points_asmr.csv", index=False)



    
def get_gpt2_model_and_tokenizer():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                return_dict_in_generate=True,
                                                pad_token_id=tokenizer.eos_token_id).to(device)
                                                
    return model_id, model, tokenizer, device



if __name__ == "__main__":
    sample_floating_point()