from tkinter import NONE
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from dataloader import DataLoader
class SVM_Text_Model:
    def __init__(self, dataloader, vocabulary=None) -> None:
        self.dataloader:DataLoader = dataloader
        self.model = svm.SVC()
        self.vocabulary = vocabulary
        self.countVectorizer = None
        self.vectorized_corpus = None
        if vocabulary is None:
            self.vectorize()
    def vectorize(self):
        if self.countVectorizer is None:
            self.countVectorizer = CountVectorizer()
        headlines = self.dataloader.get_headlines()
        self.vectorized_corpus = self.countVectorizer.fit_transform(headlines)
        self.vocabulary = self.countVectorizer.get_feature_names_out()
    def fit_model(self):
        X = self.vectorized_corpus()
        y = self.dataloader.get_labels()
        self.model.fit(X,y)
    def predict(self, text:str):
        if self.vectorized is None:
            self.vectorize()
        

    