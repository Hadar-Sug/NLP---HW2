import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class BinaryClassifier:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.model = LogisticRegression()

    def preprocess(self, words_list, tags_list=None):
        words = [word for sublist in words_list for word in sublist]
        
        # Initialize the word matrix with zeros
        word_matrix = np.zeros((len(words), self.embedding_model.vector_size))
        for i, word in enumerate(words):
            if word in self.embedding_model:
                word_matrix[i] = self.embedding_model[word]
        
        # Check if tags_list is provided
        if tags_list is not None:
            tags = [tag for sublist in tags_list for tag in sublist]
        else:
            tags = None  # Set tags to None if tags_list is not provided
        
        return word_matrix, tags


    def train(self, train_words, train_tags):
        train_embeddings, train_tags = self.preprocess(train_words, train_tags)
        self.model.fit(train_embeddings, train_tags)

    def predict(self, test_words, true_tags=None, verbose=False):
        test_embeddings, true_tags = self.preprocess(test_words,true_tags)
        predictions = self.model.predict(test_embeddings)
        
        if verbose and true_tags is not None:
            f1 = f1_score(true_tags, predictions)
            print("Logistic Regression F1 Score:", f1)
        
        return predictions
