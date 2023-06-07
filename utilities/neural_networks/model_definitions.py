from tensorflow.keras.utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import math

class Models:
    @staticmethod
    def binary_lstm_model(vocab_size, max_length, has_embedding = True, learning_rate=0.01):  
            raise NotImplementedError   
            model = Sequential()

            if(has_embedding):
                # Heuristic for embedding size according to google (https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)
                embedding_size = math.ceil(vocab_size**0.25)
                model.add(Embedding(vocab_size, embedding_size, input_length=max_length))

            model.add(LSTM(64, return_sequences=True))
            model.add(Dense(30))
            model.add(LSTM(62))
            model.add(Dense(2, activation='softmax'))
            criterion = Adam(lr=learning_rate)
            model.compile(loss='categorical_crossentropy',
                        optimizer=criterion, metrics=['accuracy'])
            model.summary()
            return model

    @staticmethod
    def last_token_lstm_model(input_vocab_size, output_vocab_size, has_embedding = True, embedding_size = None):        
            model = Sequential()

            if(has_embedding):
                # Heuristic for embedding size according to google (https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)
                if embedding_size is None:
                    embedding_size = math.ceil(input_vocab_size**0.25)
                model.add(Embedding(input_vocab_size, embedding_size, mask_zero = True)) 
            
            model.add(LSTM(64, return_sequences=True))
            model.add(Dense(30))
            model.add(LSTM(62))
            model.add(Dense(output_vocab_size, activation='softmax'))
            # criterion = Adam(lr=learning_rate)
            # model.compile(loss='categorical_crossentropy',
            #             optimizer=criterion, metrics=['accuracy'])
            # input_shape = (None, None, input_vocab_size)
            # if has_embedding:
            #     input_shape = (None, None, 1)
            # model.build(input_shape=input_shape)
            return model
    
    @staticmethod
    def time_distributed_lstm_model(vocab_size, max_length, has_embedding = True, learning_rate=0.01, lstm_units = [30]):   
            raise NotImplementedError        
            #IF has_embedding == False -> data in X will be expected to be one-hotted
            model = Sequential()

            if(has_embedding):
                # Heuristic for embedding size according to google (https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html)
                embedding_size = math.ceil(vocab_size**0.25)
                #model.add(Embedding(vocab_size, embedding_size, input_length=max_length, mask_zero = True))
                model.add(Embedding(vocab_size, embedding_size, mask_zero = True))
                vocab_size = embedding_size
            
            prev_units = vocab_size
            for units in lstm_units:
                model.add(LSTM(units, return_sequences = True, input_shape = (max_length,prev_units))) 

            model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
            criterion = Adam(lr=learning_rate)
            model.compile(loss='categorical_crossentropy',
                        optimizer=criterion, metrics=['accuracy'])
            model.summary()
            return model