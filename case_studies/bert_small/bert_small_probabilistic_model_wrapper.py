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
        return self._get_probability([str(sequence)], symbols)
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        results = []
        for seq in sequences:
            results.append(self.get_last_token_weights(seq, required_suffixes))
        return results
    
    def tokenize(self, sequence: str):
        return self.tokenizer(sequence, return_tensors="pt").to(self.device)

    def _get_probability(self, sequence: str, symbols: list):
        tokens = self.tokenize(sequence)
        token_ids = tokens.input_ids[:, :-1]

        with torch.no_grad():
            output = self.model(token_ids)
            logits = output.logits
            probs = torch.softmax(logits, dim=-1)

        word_probabilities = []
        for word in symbols:
            word_tokens = self.tokenize(str(word))

            #remove the first and last token
            word_tokens.input_ids = word_tokens.input_ids[:, 1:-1]

            word_probs = probs[0, -1, word_tokens.input_ids[-1]][0]
            amount_of_tokens_for_last_symbols = len(word_tokens.input_ids[0])
            if (amount_of_tokens_for_last_symbols > 1):
                symbol_word_probs = 1
                actual_sequence = token_ids
                with torch.no_grad():
                    for token in word_tokens.input_ids[0]:
                        new_probs = torch.softmax(self.model(actual_sequence).logits, dim=-1)
                        symbol_word_probs *= new_probs[0, -1, token]
                    actual_sequence = torch.cat((actual_sequence, token.reshape((1,1))), dim=-1)
                word_probs = symbol_word_probs
            word_probabilities.append(word_probs)
        
        return word_probabilities

    