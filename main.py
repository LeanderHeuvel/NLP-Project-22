import dataloader
from models import svm_model
from models import bert_model
import argparse

parser = argparse.ArgumentParser(description='train or evaluate a model')

parser.add_argument('--model_type', default = "bert", help = "choose between svm or bert", type=str)
parser.add_argument('--action', default = "train", help = "train or evaluate", type=str)
parser.add_argument('--model_dir', default = "checkpoints/sarcastism_ds_bert", help = "directory of model to store or load", type=str)
parser.add_argument('--epochs', default=50, type=int, help="amount of epochs for training")
parser.add_argument('--n_gram_range', default=(1,1), type=tuple, help="n-gram range of the countvectorizer for the SVM model to train on")

args = parser.parse_args()
model = None

sarcastic_loader = dataloader.DataLoader(img_dir="archive/Sarcasm_Headlines_Dataset_v2.json")
epochs = args.epochs

if args.model_type == "bert":
    model = bert_model.BertModel(dataloader = sarcastic_loader, epochs = epochs)
else:
    model = svm_model.SVM_Text_Model(sarcastic_loader, n_gram_range = args.n_gram_range)

if args.action =="train":
    print("training...")
    model.train()
    print("storing model to: "+ args.model_dir)
    model.store_model(args.model_dir)
    if args.model_type == "bert":
        model.plot_model()
        model.plot_training_curve()
else:
    print("Evaluating... ")
    model.load_model(args.model_dir)
    print(model.evaluate())

    