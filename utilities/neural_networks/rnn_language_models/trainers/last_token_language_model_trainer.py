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

from utilities.neural_networks.rnn_language_models.last_token_language_model import LastTokenLanguageModel

from utilities.symbol_encoder import SymbolEncoder
from utilities.neural_networks.model_definitions import Models

from pythautomata.utilities.sequence_generator import SequenceGenerator



class LastTokenLanguageModelTrainer(RNNLanguageModelTrainer):   
    def compile_model(self):
        criterion = self.params['criterion']
        loss = self.params['loss']
        self.model.compile(loss=loss,
                    optimizer=criterion(self.params['learning_rate']), 
                    metrics=['accuracy'])
        input_shape = (None, None, 1)
        if self._use_one_hot_on_input:
            input_shape = (None, None, len(self.alphabet)+1)
        self.model.build(input_shape=input_shape)

    def generate_target(self, transformed_data, encoder):
        X,y = transformed_data[:,:-1], transformed_data[:,-1]

        #vocab_size for y should contepmplate terminal symbols
        vocab_size = len(self.alphabet.symbols) + 1
        
        #Convert X to one-hot
        if self._use_one_hot_on_input: X = to_categorical(X, num_classes=vocab_size)

        #Convert target to one-hot (padding symbol should never be converted or contemplated as it should never be a target)
        assert encoder.encode(self.padding_symbol) not in y                
        y =  to_categorical(y-1, num_classes=vocab_size)
        
        return X,y    
  
    def _get_default_model(self):
        vocab_size = len(self.alphabet.symbols)
        input_vocab_size = vocab_size + 1
        output_vocab_size = vocab_size + 1
        model = Models.last_token_lstm_model(input_vocab_size, output_vocab_size, self.window_size-1, has_embedding=not self._use_one_hot_on_input, learning_rate=self.params['learning_rate'])
        return model
    
    def instantiate_language_model(self, encoder, model):
        return LastTokenLanguageModel(self.model_output_path, self.alphabet, encoder,
                                             model, self.window_size-1, self.padding_symbol, self.terminal_symbol, use_one_hot_on_input= self._use_one_hot_on_input)
        
    def evaluate_model(self, model, x_test, y_test):    
        ndcg = self.evaluate_ndcg_score(model, x_test, y_test)

        ndcg_per_length = dict()

        #Counting ammount of padding symbols (proxy of length)
        if self._use_one_hot_on_input:
            lengths = [self.window_size - 1 -(x[:,0]==1).sum(axis = 0) for x in x_test]
        else:
            lengths = [self.window_size - 1 -(x==0).sum(axis = 0) for x in x_test]
        zipped = list(zip(lengths, x_test, y_test))
        values = set(map(lambda x:x[0], zipped))
        newlist = [[y for y in zipped if y[0]==x] for x in values]

        for i, list_of_some_length in enumerate(newlist):
            unziped = list(zip(*list_of_some_length))
            length = unziped[0][0]        
            x_test_i = np.array(unziped[1])
            y_test_i = np.array(unziped[2])

            ndcg_per_length[length] = self.evaluate_ndcg_score(model, x_test_i, y_test_i)

        #TODO: TEST AND FIX Expected calibration error
        #expected_calibration_error = self.evaluate_expected_calibration_error(model, x_test, y_test)

        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)

        result_dict = {'ndcg_score': ndcg,
                        'ndcg_per_length': ndcg_per_length,
                        #'expected_calibration_error': expected_calibration_error, 
                        'test_loss': test_loss,
                        'test_accuracy':test_accuracy}


        return result_dict
    
    def evaluate_ndcg_score(self, model, x_test, y_test):
        test_pred = model.predict(x_test)
        vocab_size = self._get_vocab_size()

        #y_test_r = np.reshape(y_test, (-1, vocab_size)        
        #test_pred_r = np.reshape(test_pred, (-1, vocab_size)

        ndcg = ndcg_score(y_test, test_pred,k = vocab_size)  

        return ndcg
    