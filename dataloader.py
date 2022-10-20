import json
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
    def __init__(self, img_dir:str, portion=0.4 ) -> None:
        self.img_dir = img_dir
        self.portion = portion
        self.data = []
        self.headlines = None
        self.labels = None
        self.index = 0 
        self.__load_data__()
    '''
    For internal use only, loads the data after instance initialization in Python list
    '''
    def __load_data__(self):
        with open(self.img_dir) as file:
            for idx, line in enumerate(file.readlines()):
                if idx < 28618 * self.portion:
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
    def get_headlines(self):
        if self.headlines == None:
            self.headlines = [headline['headline'] for headline in self.data]
        return self.headlines
    '''
    This method loads the labels, sarcastic yes/no only. The labels are cached in the instance.
    '''
    def get_labels(self):
        if self.labels == None:
            self.labels = [label['is_sarcastic'] for label in self.data]
        return self.labels
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