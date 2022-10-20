from tkinter import NONE
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from dataloader import DataLoader

class SVM_Text_Model:
    def __init__(self, dataloader, pretrained_model = None, vocabulary=None) -> None:
        self.dataloader:DataLoader = dataloader
        if pretrained_model is None:
            self.model = svm.SVC()
        else:
            self.model = pretrained_model
        self.vocabulary = vocabulary
        self.countVectorizer = None
        self.vectorized_corpus = None
        if self.vocabulary is None:
            self.vectorize()
        
    def vectorize(self):
        if self.countVectorizer is None:
            self.countVectorizer = CountVectorizer()
        headlines = self.dataloader.get_headlines()
        self.vectorized_corpus = self.countVectorizer.fit_transform(headlines)
        self.vocabulary = self.countVectorizer.get_feature_names_out()

    def fit_model(self):
        X = self.vectorized_corpus
        y = self.dataloader.get_labels()
        self.model.fit(X,y)
    def get_vocabulary(self):
        return self.vocabulary

    def predict(self, corpus:str):
        if self.vectorized_corpus is None:
            self.vectorize()
        term_matrix = self.countVectorizer.transform(corpus)
        return self.model.predict(term_matrix)

        

    