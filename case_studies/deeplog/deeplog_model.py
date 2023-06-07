import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from pythautomata.abstract.boolean_model import BooleanModel
from pythautomata.base_types.symbol import SymbolStr
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet

class DeepLogBinaryModel(BooleanModel):

    def __init__(self, model_name: str, threshold=0.0000002, vocab_size=29, verbose = 0):
        super().__init__()
        self._model_name = model_name
        self._threshold = threshold
        self._vocab_size = vocab_size
        alphabet = []
        self._encoding_dict = {}
        for i in range(vocab_size):
            symbol = SymbolStr(str(i))
            alphabet.append(symbol) 
            self._encoding_dict[symbol] = i       
        self._alphabet = Alphabet(alphabet)
        self._verbose = verbose

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alphabet(self):
        return self._alphabet

    @property    
    def name(self) -> str:
        return self.model_name

    def accepts(self, sequence: Sequence) -> bool:
        seq = [self._encoding_dict[x] for x in sequence]
        return self.predict([seq])[0]

    def load(self):
        self._model = load_model(str(self._model_name) + '.h5')

    def get_symbol(self, symbol):
        return self._alphabet[symbol]

    def predict(self, sequences: list) -> [bool]:
        if not (self._model == None):
            proba = self._evaluate_seq(sequences[0], self._verbose)
            pred = np.array([proba > self._threshold])
            return pred
        return np.array([False])

    def _evaluate_seq(self, seq, verbose=1):
        probas = 1 if len(seq) >= 2 else 0
        batch_index = 0
        next_symbol_proba_vector = -1

        if(verbose > 0):
            print("Evaluating sequence...")
            print(seq)

        for i in range(1, len(seq)):
            if(verbose > 2):
                print("\n##################################")
                print("\nCurrent Subsequence: " + str(seq[:i]))

            expected_symbol_index = seq[i]

            if(verbose > 2):
                print("Expected Symbol: " + str(expected_symbol_index))

            onehot_seq = np.asarray([to_categorical(seq[:i], num_classes=self._vocab_size)])
            yhat = self._model.predict(onehot_seq)

            yhat_last = yhat[batch_index][next_symbol_proba_vector]
            predicted_proba_for_expected_symbol = yhat_last[expected_symbol_index]
            if(verbose > 2):
                print("Expected Symbol proba: " + str(predicted_proba_for_expected_symbol))

            probas = probas * predicted_proba_for_expected_symbol
            if(verbose > 1):
                print("Subsequence " +
                      str(seq[:i + 1]) + " proba: " + str(probas))

        if(verbose > 2):
            print("\n##################################")
        if(verbose > 0):
            print("Final proba:", str(probas), "\n\n")

        return probas
