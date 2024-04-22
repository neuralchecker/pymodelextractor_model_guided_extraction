import torch
import time
import pandas as pd
import numpy as np
import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import outlines
import outlines.caching as cache
from outlines.models.transformers import Transformers
from case_studies.gpt2.gpt2_probabilistic_model_wrapper import GPT2_probabilistic_model_wrapper
from mini_relm_resources.automata_examples.floating_point_wfa_01 import alphabet
from utilities.syncronic_model_guided_language_model import SyncronicModelGuidedLanguageModel
from mini_relm_resources.automata_examples.floating_point_wfa_01 import get_floating_point_wfa_01
from utilities.hypothesis_aware_sample_probabilistic_teacher import HypothesisAwareSampleProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.bounded_pdfa_quantization_n_ary_tree_learner import BoundedPDFAQuantizationNAryTreeLearner
from pythautomata.utilities.probability_partitioner import TopKProbabilityPartitioner, QuantizationProbabilityPartitioner, RankingPartitioner
from pythautomata.model_comparators.wfa_partition_comparison_strategy import WFAPartitionComparator
from pythautomata.utilities.guiding_wfa_sequence_generator import GuidingWDFASequenceGenerator
from pythautomata.utilities.pdfa_operations import get_representative_sample
from sampling.get_representative_sample_token import get_representative_sample_token
from sampling.get_representative_sample_length import get_representative_sample_length
from utilities.floating_point_partitioner import FloatingPointProbabilityPartitioner

# This benchmark generates every sample for each algorithm
# Watch out that outlines saves it generation in .cache, so the following generations are A LOT faster


def benchmark_algorithms(sample_size: int, number_of_executions: int = 1):
    # Make sure outlines cache is empty and disabled
    cache.clear_cache()
    cache.disable_cache()
    model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
    results = []

    for _ in range(number_of_executions):
        outlinesModel = Transformers(model, tokenizer)

        # ------------------------------------- OUTLINES --------------------------------------------
        prompt = tokenizer.decode(tokenizer.bos_token_id)

        gen_time = 0
        start_time = time.time()
        outlinesGenerator = outlines.generate.regex(outlinesModel, "\.[0-9]")
        gen_time = time.time() - start_time
        sample_time = 0
        start_time = time.time()
        for i in range(sample_size):
            _ = outlinesGenerator(prompt)
        sample_time = time.time() - start_time
        
        

        res = ("Outlines", sample_size, gen_time, sample_time)
        results.append(res)
        print(res)
        
        # ------------------------------------- PDFA --------------------------------------------

        model_id, model, tokenizer, device = get_gpt2_model_and_tokenizer()
        wrapper = GPT2_probabilistic_model_wrapper(50, alphabet, device, model, tokenizer)
        property_model = get_floating_point_wfa_01(wrapper.terminal_symbol)
        synchronic_model = SyncronicModelGuidedLanguageModel(wrapper, property_model, model_name="GUIDED_GPT2", max_seq_length=10, normalize_outputs=True)
        guiding_generator = GuidingWDFASequenceGenerator(property_model, None)
        partitioner = FloatingPointProbabilityPartitioner()
        comparator = WFAPartitionComparator(partitioner)
        epsilon = 0.05
        delta = epsilon
        sequence_generator = guiding_generator
        max_states = 100
        max_query_length = 100
        teacher = HypothesisAwareSampleProbabilisticTeacher(synchronic_model, comparator = comparator, max_seq_length = 4, sample_size = 100)
        learner = BoundedPDFAQuantizationNAryTreeLearner(partitioner = partitioner, max_states = max_states, max_query_length = max_query_length, max_seconds_run = None, generate_partial_hipothesis = True, pre_cache_queries_for_building_hipothesis = True,  check_probabilistic_hipothesis = True, mean_distribution_for_partial_hipothesis = False, omit_zero_transitions = True)

        
        gen_time = 0
        start_time = time.time()
        learning_result = learner.learn(teacher, verbose=False)
        gen_time = time.time() - start_time
        pdfa = learning_result.model

        sample_time = 0
        start_time = time.time()
        #get_representative_sample(pdfa, sample_size)
        get_representative_sample_length(pdfa, sample_size = sample_size, length=1, retry=False)
        sample_time = time.time() - start_time
        
        res = ("PDFA", sample_size, gen_time, sample_time)
        results.append(res)
        print(res)
        
        # ------------------------------------- GPT2 --------------------------------------------

        gen_time = 0 
        sample_time = 0
        start_time = time.time()

        for i in range(sample_size):

            next_token = ""
            prompt = [tokenizer.decode(tokenizer.bos_token_id) , "."]
            min_digits = 1
            max_digits = 1
            while next_token != tokenizer.decode(tokenizer.eos_token_id):            
                if len(prompt) > min_digits+1:
                    normalized_word_probs = calculate_probs(prompt, True, tokenizer, device, model)
                else:
                    normalized_word_probs = calculate_probs(prompt, False, tokenizer, device, model)      

                next_token = np.random.choice(a=list(normalized_word_probs), p=list(normalized_word_probs.values()))
                if next_token != tokenizer.decode(tokenizer.eos_token_id):
                    prompt.append(next_token)                        
                if len(prompt)>=max_digits:
                    next_token = tokenizer.decode(tokenizer.eos_token_id)
        sample_time = time.time() - start_time

        res = ("GPT2", sample_size, gen_time, sample_time)
        results.append(res)
        print(res)

    # dfresults = pd.DataFrame(results, columns=["Algorithm", "Samples", "Generation Time", "Sample Time"])
    # try:
    #      os.mkdir("./benchmarks/results")
    # except OSError as error:
    #      print(error)
    # dfresults.to_csv(f"./benchmarks/results/benchmark_1_{time.time()}.csv", index=False)

    
    
def calculate_probs(prompt, eos, tokenizer, device, model):
    
    str_seq = [tokenizer.tokenize(x) for x in prompt]
    str_seq = [item for tokens in str_seq for item in tokens]
    prompt_ids = tokenizer.convert_tokens_to_ids(str_seq)        
    input_ids = torch.tensor(prompt_ids).reshape(1, -1)  
    with torch.no_grad():
            output = model(input_ids)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]
 
 
    numbers = ["0", "1", "2","3","4","5","6","7","8","9"]
    indexes = [tokenizer.encode(number) for number in numbers]
    if eos:
        indexes.append([tokenizer.eos_token_id])
    word_probs = {}
    for i in indexes:
        word_prob = probs[i]
        word_probs[tokenizer.decode(i).replace(" ","")] = word_prob.item()
    normalized_word_probs = {}
    total = sum(word_probs.values())
    for word in word_probs:
        normalized_word_probs[word] = word_probs[word] / total
    return normalized_word_probs

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
    benchmark_algorithms()
