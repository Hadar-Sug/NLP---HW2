from itertools import chain
import pickle
import random
import torch
import numpy as np
import gensim.downloader as api
import os
from LogisticRegressor import BinaryClassifier
from FFNNLSTM import NeuralNetworks

# import fasttext.util 
def random_permutations(x_train, y_train):
    
    # Get the number of samples
    num_samples = len(x_train)

    # Create shuffled indices
    shuffled_indices = np.random.permutation(num_samples)

    # Use shuffled indices to rearrange both X and Y
    x_shuffled = [x_train[i] for i in shuffled_indices]
    y_shuffled = [y_train[i] for i in shuffled_indices]
    return x_shuffled, y_shuffled

def flatten (list_of_lists):
    return list(chain.from_iterable(list_of_lists))

def parse_tagged_data_binary(filename):
    """
    Parses the NER dataset file, handling different formats.
    Assumes sentences are separated by empty lines and each line contains a word and its tag separated by a tab.
    """
    with open(filename, 'r') as file:
        sentences, tags = [], []
        current_sentence, current_tags = [], []
        for line in file:
            if line.strip():  # Non-empty line
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    word, tag = parts
                    current_sentence.append(word)
                    
                    # Convert non-'O' tags to 'ENT'
                    tag = 'ENT' if tag != 'O' else tag
                    current_tags.append(tag)
                else:
                    # Print and skip the line with unexpected format
                    # print(f"Skipping line due to unexpected format: {line.strip()}")
                    continue  # Skip this iteration
            else:  # Sentence boundary
                if current_sentence:
                    sentences.append(current_sentence)  # Add the sentence as a list of words
                    tags.append(current_tags)  # Add the corresponding tags
                    current_sentence, current_tags = [], []  # Reset for next sentence
        # Handle last sentence if file does not end with an empty line
        if current_sentence:
            sentences.append(current_sentence)  # Add the last sentence
            tags.append(current_tags)  # Add the last set of tags
    return sentences, tags

def parse_test_data(file_path):
    """
    Parses the test dataset file.
    Assumes each line contains only a word.
    """
    test_sentences = []
    with open(file_path, 'r') as f:
        current_sentence = []
        for line in f:
            word = line.strip()
            if word:
                current_sentence.append(word)
            else:
                if current_sentence:
                    test_sentences.append(current_sentence)
                    current_sentence = []
        if current_sentence:  # Add the last sentence if present
            test_sentences.append(current_sentence)
    return test_sentences

def add_predictions_to_test_file(sentences, output_file_path, tags):
  with open(output_file_path, 'w') as f:
      for sentence, sentence_tags in zip(sentences, tags):
          for word, tag in zip(sentence, sentence_tags):
              f.write(f"{word}\t{tag}\n")  # Write word and tag separated by a tab
          f.write("\n")  # Separate sentences with a blank line

def convert_to_binary(c_list, ix_to_tag):
    """
    Converts a list of categorical indices into a binary list.
    
    Args:
    - c_list: List of categorical indices (e.g., [1, 2, 0, 1]).
    - ix_to_tag: Mapping from indices to tags (e.g., {0: 'O', 1: 'B-LOC', 2: 'I-PER'}).
    
    Returns:
    - A list where each element is 'O' if the corresponding tag in ix_to_tag is 'O',
      otherwise 'ENT'.
    """
    binary_list = ['O' if ix_to_tag[ix] == 'O' else 'ENT' for ix in c_list]
    return binary_list


def lower_case_sentences(list_of_lists_of_words):
    """
    Lowercase words in a list of lists of words based on corresponding tags.
    Words are only converted to lowercase if their corresponding tag is 'O'.

    Args:
        list_of_lists_of_words (list of lists of str): The input list of lists of words.
        list_of_lists_of_tags (list of lists of str): The input list of lists of tags corresponding to the words.

    Returns:
        list of lists of str: The list of lists of words with words converted to lowercase based on corresponding tags.
    """
    return [[word.lower() for word in words] for words in list_of_lists_of_words]


def save_to_pickle(data, directory, filename):
    """
    Saves the given data to a pickle file in the specified directory with the given filename.
    """
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")


def read_pickle_file(directory, filename):
    """
    Read a pickle file and return its content.
    
    Args:
    - directory (str): The directory path where the pickle file is located.
    - filename (str): The name of the pickle file.
    
    Returns:
    The content of the pickle file.
    """
    with open(os.path.join(directory, filename), 'rb') as f:
        data = pickle.load(f)
    return data


def main():    
    np.random.seed(6942)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_directory = os.path.dirname(os.path.realpath(__file__))
    # # Define paths for training, validation, and test files
    train_file_path = os.path.join(current_directory, "data", "train.tagged")
    validation_file_path = os.path.join(current_directory, "data", "dev.tagged")
    test_file_path = os.path.join(current_directory, "data", "test.untagged")
    train_sentences, train_tags = parse_tagged_data_binary(train_file_path)
    val_sentences, val_tags = parse_tagged_data_binary(validation_file_path)
    test_sentences = parse_test_data(test_file_path)
    actual_training_data, actual_training_tags = random_permutations(train_sentences, train_tags)
    glove_model = api.load("glove-twitter-50")
    word2vec = api.load("word2vec-google-news-300")
    tag_to_ix = {'O': 0, 'ENT': 1}
    # Create the reverse mapping
    ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
    unique_words = set(word for sentence in actual_training_data+val_sentences for word in sentence)
    submission = True 
    if not submission:
        # model 1 - Logistic regression
        binary_training_tags = [[tag_to_ix[tag] for tag in tag_seq] for tag_seq in train_tags]
        binary_val_tags = [[tag_to_ix[tag] for tag in tag_seq] for tag_seq in val_tags]
        LogisticRegression = BinaryClassifier(word2vec)
        LogisticRegression.train(train_sentences, binary_training_tags)
        LogisticRegression.predict(train_sentences, binary_training_tags, verbose=True)
        LogisticRegression.predict(val_sentences, binary_val_tags, verbose=True)
        
        nn = NeuralNetworks(128,glove_model,tag_to_ix,device,unique_words,word2vec,ix_to_tag,val_sentences,val_tags,"FFNN")
        
        # model 2 - FFNN 
        nn.train(actual_training_data, actual_training_tags,"FFNN",num_epochs=5)
        dev_score = nn.predict(val_sentences, val_tags,verbose=True)
        train_score = nn.predict(train_sentences, train_tags,verbose=True)
        print("Dev F1 Score FFNN:", dev_score)
        print("Train F1 Score FFNN :", train_score)
        
        # model 3 - LSTM
        nn.train(actual_training_data, actual_training_tags,"LSTM",num_epochs=7)
        dev_score = nn.predict(val_sentences, val_tags,verbose=True)
        train_score = nn.predict(train_sentences, train_tags,verbose=True)    
        print("Dev F1 Score LSTM:", dev_score)
        print("Train F1 Score LSTM :", train_score)
    
    # comp
    #training on all the data
    train_val_sentences, train_val_tags = train_sentences + val_sentences, train_tags + val_tags
    actual_training_data, actual_training_tags = random_permutations(train_val_sentences, train_val_tags)
    unique_words = set(word for sentence in actual_training_data+val_sentences+test_sentences for word in sentence)
    
    nn = NeuralNetworks(128,glove_model,tag_to_ix,device,unique_words,word2vec,ix_to_tag,train_val_sentences,train_val_tags,"COMP")
    nn.train(actual_training_data, actual_training_tags,"COMP",num_epochs=7) 
    predictions = nn.predict(test_sentences)
    add_predictions_to_test_file(test_sentences, "comp_206567067_318155843.tagged", predictions)
    

if __name__ == "__main__":
    main()