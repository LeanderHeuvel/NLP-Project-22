import json
from sklearn.model_selection import train_test_split
'''
Dataloader of Sarcastic dataset

Accepts data with the following format

{'headline': ... , 'is_sarcastic': ...}

This Dataloader is iterable.

by Leander van den Heuvel

'''
class DataLoader:
    '''
    Load data in python list and lazy load the headlines and labels in seperate list. 
    '''
    def __init__(self, img_dir:str, portion=0.4, train_test_split=True ) -> None:
        self.img_dir = img_dir
        self.portion = portion
        self.data = []
        self.headlines = None
        self.labels = None
        self.index = 0 
        self.random_state = 9283
        if train_test_split:
            self.train = []
            self.test = []
            self.X_train = []
            self.X_test = []
            self.y_train = []
            self.y_test = []
        self.__load_data__()
        self.__split_data__(train_test_split)
    '''
    For internal use only, loads the data after instance initialization in Python list
    '''
    def __load_data__(self):
        with open(self.img_dir) as file:
            for idx, line in enumerate(file.readlines()):
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    '''
    Given a valid index number, this method returns the corresponding element in the dataset
    '''
    def get_item(self, idx:int) :
        if idx < len(self) or idx >= 0:
            headline:str = self.data[idx]['headline']
            is_sarcastic:int = self.data[idx]['is_sarcastic']
            return headline, is_sarcastic
        else:
            raise IndexError
    '''
    This method loads the headlines only. The headlines are cached in the instance. Can be useful for a countvectorizer for example.
    '''
    def __split_data__(self, split):
        if split:
            data = train_test_split(self.data,random_state = self.random_state, train_size=0.2)
            self.train = data[0]
            self.test = data[1]
        else:
            self.train = self.data
            self.test = self.data

    def get_training_data(self):   
        if self.X_train == []:
            self.X_train = [element['headline'] for element in self.train]
            self.y_train = [element['is_sarcastic'] for element in self.train]
        return self.X_train, self.y_train

    '''
    This method loads the labels, sarcastic yes/no only. The labels are cached in the instance.
    '''
    def get_test_data(self):
        if self.X_test == None:
            self.y_test = [element['is_sarcastic'] for element in self.test]
            self.X_test = [element['headline'] for element in self.test]
        return self.X_test, self.y_train
    
    '''
    Iterable implementation.
    '''
    def __next__(self) :
        if self.index < len(self):
            headline:str = self.data[self.index]['headline']
            is_sarcastic:int = self.data[self.index]['is_sarcastic']
            self.index += 1
            return headline, is_sarcastic
        else:
            raise StopIteration