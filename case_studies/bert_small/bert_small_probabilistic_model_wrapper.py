from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
import torch
import numpy as np

class BERT_SMALL_probabilistic_model_wrapper(ProbabilisticModel):

    def __init__(self, max_length, alphabet: Alphabet, device: str, model, tokenizer):
        self.model = model 
        self.device = device
        self.tokenizer = tokenizer
        self._alphabet = alphabet

    @property
    def name(self) -> str:
        return "BERT_SMALL"
    
    @property
    def terminal_symbol(self) -> Symbol:
        return SymbolStr(self.tokenizer.sep_token)
    
    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet
    
    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
    
    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def last_token_probability(self, sequence: Sequence) -> float:
        alphabet_symbols = list(self.alphabet.symbols)
        weights = self.guide_model.get_last_token_weights(sequence, alphabet_symbols)
        symbols = [str(alphabet_symbols[i]).split(" ")[0] for i in range(len(weights)) if weights[i] > 0]

        if not len(symbols):
            symbols.add(self.terminal_symbol)
        
        prob = self._get_probability(str(sequence).replace(",", " "), symbols)
        return prob
    
    def get_last_token_weights(self, sequence, symbols):
        return [weight for weight in self._get_probability([str(sequence)], symbols, False)]
    
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

        word_probabilities = []
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
            word_probabilities.append(total_word_probs.item())


        if normalize:
            words = np.array(word_probabilities)
            words /= np.sum(words)
            word_probabilities = words.tolist()

        return word_probabilities

    