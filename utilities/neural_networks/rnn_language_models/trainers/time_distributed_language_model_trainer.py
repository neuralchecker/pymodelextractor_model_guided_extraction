#from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

from utilities.neural_networks.rnn_language_models.trainers.rnn_language_model_trainer import RNNLanguageModelTrainer

import math
import numpy as np

from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence

from utilities.neural_networks.rnn_language_models.time_distributed_language_model import TimeDistributedLanguageModel

from utilities.symbol_encoder import SymbolEncoder
from utilities.neural_networks.model_definitions import Models

from pythautomata.utilities.uniform_length_sequence_generator import UniformLengthSequenceGenerator



class TimeDistributedLanguageModelTrainer(RNNLanguageModelTrainer):  
    def generate_target(self, transformed_data, encoder):
        X,y = transformed_data[:,:-1], transformed_data[:,1:]

        #vocab_size should contepmplate padding and terminal symbols
        vocab_size = self._get_vocab_size()

        #Convert to one-hot
        if self.use_one_hot: X = to_categorical(X, num_classes=vocab_size)
        y =  to_categorical(y, num_classes=vocab_size)

        return X,y    

    def _get_default_model(self):
        vocab_size = self._get_vocab_size()
        model = Models.time_distributed_lstm_model(vocab_size, self.window_size-1)
        return model
    
    def instantiate_language_model(self, encoder, model):
        return TimeDistributedLanguageModel(self.model_output_path, self.alphabet, encoder,
                                             model, self.window_size-1, self.padding_symbol, self.terminal_symbol)
        
    def evaluate_model(self, model, x_test, y_test):   
        test_pred = model.predict(x_test)
        vocab_size = self._get_vocab_size()
        y_test_r = np.reshape(y_test, (-1, vocab_size))        
        test_pred_r = np.reshape(test_pred, (-1, vocab_size))

        ndcg = ndcg_score(y_test_r, test_pred_r,k = vocab_size)  

        result_dict = {'ndcg_score': ndcg}
        return result_dict