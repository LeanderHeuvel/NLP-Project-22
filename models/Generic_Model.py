import abc
from dataloader import DataLoader

class GenericModelInterface(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def store_model(self, model_dir):
        pass
    @abc.abstractmethod
    def load_model(self, model_dir):
        pass
    @abc.abstractmethod
    def predict(self, corpus)-> int:
        pass
    @abc.abstractmethod
    def evaluate(self) -> float:
        pass