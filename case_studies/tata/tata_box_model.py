from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from neural_networks.binary_rnn_acceptors.nn_binary_classifier_model import NNBinaryClassifierModel
import numpy as np


class TATABoxModel(NNBinaryClassifierModel):

    def __init__(self, model_name: str, threshold=0.5, max_seq_length=50):
        super().__init__()
        self._model_name = model_name
        self._threshold = threshold
        self._alphabet = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        self._max_seq_length = max_seq_length

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alphabet(self):
        return self._alphabet.keys()

    def load(self):
        self._model = load_model(str(self._model_name) + '.h5')

    def get_symbol(self, symbol):
        return self._alphabet[symbol]

    def predict(self, sequences: list) -> [bool]:
        if not (self._model == None):
            if len(sequences[0]) < self._max_seq_length:
                proba = np.array([0])
            else:
                sequence = sequences[0][0: self._max_seq_length]
                sequence_oh = np.asarray([to_categorical(sequence, num_classes=len(self._alphabet))])
                proba = self._model.predict(sequence_oh)[0]

            pred = proba > self._threshold
            return pred
        return np.array([False])
