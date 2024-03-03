import torch
import time
import pandas as pd
import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import outlines
from outlines.models.transformers import Transformer, TransformerTokenizer
from case_studies.gpt2.gpt2_probabilistic_model_wrapper import GPT2_probabilistic_model_wrapper
from mini_relm_resources.automata_examples.man_woman_wfa import alphabet
from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from mini_relm_resources.automata_examples.man_woman_wfa import get_man_woman_wfa
from pymodelextractor.teachers.pac_probabilistic_teacher import PACProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner, QuantizationProbabilityPartitioner, RankingPartitioner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator
from pythautomata.utilities.pdfa_operations import get_representative_sample

# This benchmarks generate once for each algorithm and then samples 

def benchmark_algorithms(number_of_executions: int = 1, samples: list[int] = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]):
    # TODO probably want to receive the model as a parameter
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    
    # Build the custom outlinesModel using the transformer model and tokenizer
    outlinesModel = Transformer(model, TransformerTokenizer(model_id))

    results = []
    
    outlines_gen_time = 0
    start_time = time.time()
    # TODO check that this regex equals the regex used in mini relm
    outlinesGenerator = outlines.generate.regex(outlinesModel, "The (man|woman) was trained in (medicine|science|engineering|maths|art|music|astrophysics|astrology)")
    outlines_gen_time = time.time() - start_time
    
    wrapper = GPT2_probabilistic_model_wrapper(50, alphabet, device, model, tokenizer)
    property_model = get_man_woman_wfa(wrapper.terminal_symbol)
    syncrhronic_model = SyncronicModelGuidedLanguageModel(wrapper, property_model, model_name="GUIDED_GPT2", max_seq_length=10,normalize_outputs=True, top_k=2)
    partitioner = QuantizationProbabilityPartitioner(100000)
    guiding_generator = GuidingWDFASequenceGenerator(property_model, None)
    comparator = WFAPartitionComparator(partitioner)
    epsilon = 0.1
    delta = epsilon
    max_states = 30
    max_query_length = 100
    teacher  = PACProbabilisticTeacher(syncrhronic_model, epsilon = epsilon, delta = delta, max_seq_length = None, comparator = comparator, sequence_generator=guiding_generator, compute_epsilon_star=False)
    learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner, max_states, max_query_length, None, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True,  check_probabilistic_hipothesis = False, omit_zero_transitions=True)
            
    asmr_gen_time = 0
    start_time = time.time()
    learning_result = learner.learn(teacher, verbose=False)
    asmr_gen_time = time.time() - start_time
    pdfa = learning_result.model

    for sample in samples:
        for _ in range(number_of_executions):
            # Outlines
            prompt = " "
            sample_time = 0
            start_time = time.time()
            for i in range(sample):
                _ = outlinesGenerator(prompt)
            sample_time = time.time() - start_time

            res = ("Outlines", sample, outlines_gen_time, sample_time)
            results.append(res)
            print(res)
            
            # ASMR?
            

            sample_time = 0
            start_time = time.time()
            get_representative_sample(pdfa, sample)
            sample_time = time.time() - start_time
            
            res = ("ASMR", sample, asmr_gen_time, sample_time)
            results.append(res)
            print(res)

    dfresults = pd.DataFrame(results, columns=["Algorithm", "Samples", "Generation Time", "Sample Time"])
    try:
         os.mkdir("./benchmarks/results")
    except OSError as error:
         print(error)
    dfresults.to_csv(f"./benchmarks/results/benchmark_1_{time.time()}.csv", index=False)

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
    benchmark_algorithms()