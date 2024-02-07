from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
import torch

class BERT_SMALL_probabilistic_model_wrapper(ProbabilisticModel):

    def __init__(self, max_seq_length: int, alphabet:Alphabet, device: str, model, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.device = device
        self._alphabet = alphabet

    @property
    def name(self) -> str:
        return "BERT_SMALL"
    
    @property
    def terminal_symbol(self) -> Symbol:
        print(self.tokenizer.sep_token)
        return SymbolStr(self.tokenizer.eos_token)
    
    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
    
    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
    

    def last_token_probability(self, sequence: Sequence) -> float:
        symbols = set(self._alphabet.symbols)
        symbols.add(self.terminal_symbol)
        return self._get_probability(sequence, symbols)
    

    #TODO: Fix interface, this should be removed from the learners and pymodelextractor as a whole
    def get_last_token_weights(self, sequence, required_suffixes):
        weights = list()
        #alphabet_symbols_weights = self.next_symbol_probas(sequence)
        #alphabet_symbols_weights = {Sequence() + k: alphabet_symbols_weights[k] for k in alphabet_symbols_weights.keys()}
        alphabet_symbols_weights = self.last_token_probability(sequence)
        for suffix in required_suffixes:
            if suffix in alphabet_symbols_weights:
                weights.append(alphabet_symbols_weights[suffix])
            else:
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
    
    def tokenize(self, sequence: str):
        return self.tokenizer(sequence, return_tensors="pt").to(self.device)

    def _get_probability(self, sequence: str, symbols: list, normalize: bool = True):
        tokens = self.tokenize(sequence)
        with torch.no_grad():
            output = self.model(tokens.input_ids)
            logits = output.logits
            probs = torch.softmax(logits, dim=-1)

        word_probabilities = {}
        for word in symbols:
            word_tokens = self.tokenize(str(word))
            
            #transform tensor to list
            word_tokens.input_ids = word_tokens.input_ids.tolist()

            #remove the first and last token
            word_tokens.input_ids = [i[1:-1] for i in word_tokens.input_ids]

            #get the probability of the word
            word_probs = probs[0, -1, word_tokens.input_ids[-1]]

            total_word_probs = sum(word_probs)
            total_word_probs /= len(word_probs)
            word_probabilities[word] = total_word_probs.item()


        if normalize:
            total = sum(word_probabilities.values())
            for word in word_probabilities:
                word_probabilities[word] /= total


        return word_probabilities