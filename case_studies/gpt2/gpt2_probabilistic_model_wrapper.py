

from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from collections import OrderedDict
import torch

class GPT2_probabilistic_model_wrapper(ProbabilisticModel):
    
    def __init__(self, max_seq_length: int, alphabet:Alphabet, device: str, model, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.device = device
        self._alphabet = alphabet

    @property
    def name(self) -> str:
        return "GPT_2"
    
    @property
    def terminal_symbol(self) -> Symbol:
        return SymbolStr(self.tokenizer.eos_token)

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
    
    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    
    def last_token_probability(self, sequence: Sequence, symbols = None) -> float:
        if symbols is None:
            symbols = set(self._alphabet.symbols)
            symbols.add(self.terminal_symbol)
        return self._get_probability(sequence, symbols)
    
    #TODO: Fix interface, this should be removed from the learners and pymodelextractor as a whole
    def get_last_token_weights(self, sequence, required_suffixes):
        weights = list()
        #alphabet_symbols_weights = self.next_symbol_probas(sequence)
        #alphabet_symbols_weights = {Sequence() + k: alphabet_symbols_weights[k] for k in alphabet_symbols_weights.keys()}
        alphabet_symbols_weights = self.last_token_probability(sequence, required_suffixes)
        for suffix in required_suffixes:
            if suffix in alphabet_symbols_weights:
                weights.append(alphabet_symbols_weights[suffix])
            else:
                assert False
                new_sequence = sequence + suffix
                new_prefix = Sequence(new_sequence[:-1])
                new_suffix = new_sequence[-1]
                next_symbol_weights = self.last_token_probability(new_prefix)
                weights.append(next_symbol_weights[new_suffix])
        return weights  
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        results = []
        for seq in sequences:
            results.append(self.get_last_token_weights(seq, required_suffixes))
        return results
    
    def build_ids_sequence_from_tokens(self, sequence):
        return torch.tensor([self.tokenizer.bos_token_id,] + self.tokenizer.convert_tokens_to_ids(sequence)).reshape(1, -1).to(self.device)

    def tokenize_empty(self):
        return torch.tensor([self.tokenizer.bos_token_id,]).reshape(1, -1).to(self.device)
    
    def _get_probability(self, sequence, symbols):#, normalize, top_k=None):
        if len(sequence) == 0:
            input_ids = self.tokenize_empty()
        else:
            str_seq = [self.tokenizer.tokenize(str(x)) for x in sequence]
            str_seq = [item for tokens in str_seq for item in tokens]
            input_ids = self.build_ids_sequence_from_tokens(str_seq)
        
        with torch.no_grad():
            output = self.model(input_ids)
            # logits = output[0]
            # probs = logits.softmax(-1)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            # if top_k is not None:
            #     _axis = len(probs.shape) - 1
            #     top_k_val = torch.topk(probs, axis=_axis, k=top_k)
            #     probs[:] = 0.
            #     probs = probs.scatter(_axis,
            #                     top_k_val.indices,
            #                     top_k_val.values)
            #     probs /= torch.sum(probs)

        return self._get_symbols_probabilities_dict(input_ids, probs, symbols)#, normalize)

    # TODO: We should make sure that we are calculating the probabilities for the correct words
    # Since the tokenizer splits words in different ways, we should check that the probabilities
    # make sense
    def _get_symbols_probabilities_dict(self, input_ids, probs, symbols):#, normalize):
        symbols_probabilities = {}
        for symbol in symbols:
            #tokenizer.encode = tokenizer.tokenize + tokenizer.convert_tokens_to_ids
            tokens = self.tokenizer.tokenize(str(symbol))
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            symbol_prob = probs[0][token_ids[0]]
            if len(token_ids) > 1:   
                # Extract probabilities for the specified word from the distribution of the next token
                for i,id in enumerate(token_ids[:-1]):
                    input_ids = torch.cat([input_ids[0], torch.tensor([id])]).unsqueeze(0)
                    with torch.no_grad():
                        output = self.model(input_ids)
                        logits = output.logits[:, -1, :]
                        next_probs = torch.softmax(logits, dim=-1)
                    symbol_prob *= next_probs[0][token_ids[i+1]]
            symbols_probabilities[symbol] = symbol_prob
        symbols_probabilities = OrderedDict(symbols_probabilities)    
        return symbols_probabilities
            
        # Normalize the probabilities
        #if normalize:
        #    total = sum(word_probabilities.values())
        #    for word in word_probabilities:
        #        word_probabilities[word] /= total

        