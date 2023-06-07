from utilities.neural_networks.rnn_language_models.trainers.time_distributed_language_model_trainer import TimeDistributedLanguageModelTrainer
from utilities.neural_networks.rnn_language_models.trainers.last_token_language_model_trainer import LastTokenLanguageModelTrainer
from pythautomata.utilities.sequence_generator import SequenceGenerator
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import ProbabilisticDeterministicFiniteAutomaton as PDFA
from pythautomata.utilities.pdfa_operations import get_representative_sample

# def train_binary_network(target_model, max_seq_length, training_data_size, seed, output_path, padding_symbol, patience = 5, epochs= 30, batch_size= 30, learning_rate = 0.01):
#         if padding_symbol in target_model.alphabet.symbols:
#             raise ValueError('Padding symbol can\'t belong to target_model alphabet')

#         trainer = FrameworkBinaryNeuralNetworkTrainer(output_path)

#         generator = SequenceGenerator(target_model.alphabet, max_seq_length, seed)

#         data_x = []
#         data_y = []
#         for word in generator.generate_words(training_data_size):
#             data_x.append(generator.pad(word, padding_symbol))
#             data_y.append(target_model.accepts(word))

#         alphabet = target_model.alphabet
#         return trainer.train_network(data_x, data_y, alphabet, padding_symbol, max_seq_length, patience = patience, epochs = epochs, batch_size = batch_size, learning_rate = learning_rate)

def validate_params_and_generate_data(target_model, max_seq_length, generated_training_data_size, window_size, seed, padding_symbol, terminal_symbol):
    if padding_symbol in target_model.alphabet.symbols:
            raise ValueError('Padding symbol can\'t belong to target_model alphabet')
        
    if terminal_symbol in target_model.alphabet.symbols:
        raise ValueError('Terminal symbol can\'t belong to target_model alphabet')

    if isinstance(target_model, PDFA):
        return get_representative_sample(target_model, generated_training_data_size)

    else:
        generator = SequenceGenerator(target_model.alphabet, max_seq_length, seed)
        words = generator.generate_words(generated_training_data_size)
        #Keep only sequences belonging to target model
        data = list(filter(lambda x: target_model.accepts(x), words))
        return data

def generate_data_and_train_time_distributed_language_model_network(target_model, max_seq_length, generated_training_data_size, window_size, seed, output_path, padding_symbol, terminal_symbol, model, params, has_embedding = True):
        data = validate_params_and_generate_data(target_model, max_seq_length, generated_training_data_size, window_size, seed, padding_symbol, terminal_symbol)
        alphabet = target_model.alphabet
        trainer = TimeDistributedLanguageModelTrainer(output_path, data, window_size, alphabet, padding_symbol, terminal_symbol, has_embedding = has_embedding, model=model, params = params, seed = seed)
        model, evaluation = trainer.train_network()
        return model, evaluation

def generate_data_and_train_last_token_language_model_network(target_model, max_seq_length, generated_training_data_size, window_size, seed, output_path, padding_symbol, terminal_symbol, model, params, has_embedding = True):
        data = validate_params_and_generate_data(target_model, max_seq_length, generated_training_data_size, window_size, seed, padding_symbol, terminal_symbol)
        alphabet = target_model.alphabet
        trainer = LastTokenLanguageModelTrainer(output_path, data, window_size, alphabet, padding_symbol, terminal_symbol, has_embedding = has_embedding, model=model, params = params, seed = seed)
        model, evaluation = trainer.train_network()
        return model, evaluation

#Needs testing
def load_data_and_train_last_token_language_model_network(data_loader, output_path, window_size, padding_symbol, terminal_symbol, model, params, seed, has_embedding = True):
        data = data_loader.data
        alphabet = data_loader.alphabet
        trainer = LastTokenLanguageModelTrainer(output_path, data, window_size, alphabet, padding_symbol, terminal_symbol, has_embedding = has_embedding, model=model, params = params, seed = seed)
        model, evaluation = trainer.train_network()
        return model, evaluation

#Needs testing
def load_data_and_train_time_distributed_language_model_network(data_loader, output_path, window_size, padding_symbol, terminal_symbol, model, params, seed, has_embedding = True):        
        data = data_loader.data
        alphabet = data_loader.alphabet
        trainer = TimeDistributedLanguageModelTrainer(output_path, data, window_size, alphabet, padding_symbol, terminal_symbol, has_embedding = has_embedding, model=model, params = params, seed = seed)
        model, evaluation = trainer.train_network()
        return model, evaluation        

def generate_data_and_train_binary_classifier_network():
        raise NotImplemented