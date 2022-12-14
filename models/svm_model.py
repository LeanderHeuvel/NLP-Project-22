import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, auc, precision_score, recall_score

from dataloader import DataLoader
from models.generic_model import GenericModelInterface

from joblib import dump, load

class SVM_Text_Model(GenericModelInterface):
    
    def __init__(self, dataloader, pretrained_model = None, vocabulary=None, n_gram_range:tuple = (1,1)) -> None:
        self.dataloader:DataLoader = dataloader
        if pretrained_model is None:
            self.model = svm.SVC()
        else:
            self.model = pretrained_model
        self.vocabulary = vocabulary
        self.X_train, self.y_train = self.dataloader.get_training_data()
        self.countVectorizer = None
        self.n_gram_range = tuple(n_gram_range)
        if self.vocabulary is None:
            self.vectorize(self.X_train)
        
    def vectorize(self, X):
        if self.countVectorizer is None:
            self.countVectorizer = CountVectorizer(ngram_range = self.n_gram_range)
            self.vectorized_corpus = self.countVectorizer.fit_transform(self.X_train)
            self.vocabulary = self.countVectorizer.get_feature_names_out()
        return self.countVectorizer.transform(X)

    def train(self):
        self.model.fit(self.vectorized_corpus,self.y_train)
    
    def get_model(self):
        return self.model

    def get_vocabulary(self):
        return self.vocabulary

    def store_vocabulary(self, voc_dir):
        with open(voc_dir, "w") as out_file:
            out_file.write("\n".join(self.vocabulary))

    def load_vocabulary(self, voc_dir):
        with open(voc_dir,"r") as in_file:
            vocabulary = in_file.read().splitlines()
        self.countVectorizer = CountVectorizer(vocabulary=vocabulary)

    def predict(self, corpus):
        term_matrix = self.vectorize(corpus)
        return self.model.predict(term_matrix)

    def load_model(self, model_dir):
        self.model = load(model_dir)

    def store_model(self, model_dir):
        dump(self.model, model_dir) 
    
    def get_accuracy(self, y_test, y_pred):
        return np.where(np.array(y_test)==np.array(y_pred))[0].shape[0]/np.array(y_test).shape[0]

    def evaluate(self):
        X_test, y_test = self.dataloader.get_test_data()
        y_pred = self.predict(X_test)
        return self.get_accuracy(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), auc(y_test, y_pred)