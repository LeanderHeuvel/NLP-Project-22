# NLP-Project-22
*By Jasper Peetsma and Leander van den Heuvel*

This repository contains the project code for the "Natural Language Processing" course. In this project we study the effects of additional context on the predicitve performance of machine learning models for sarcasm detection. Currently, the project compares the following models:

 - BERT small, Uncased L-4 H-512 A-8 (can be changed in the code)
 - Support Vector Machine in combination with a countvectorizer (n-gram range = 1).

After cloning the repository the code can be ran by issuing the following command:
 
```
python3 main.py --model_type "bert" --action "train" --model_dir "checkpoints/" --epochs 50
```

main.py takes the following arguments:

 - model_type: either bert or svm
 - action: train or evaluate
 - model_dir: directory to load or save model, depending on the chosen action
 - epochs: if train, amount of epochs can be specified here