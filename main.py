from sklearn import svm
import dataloader
from models import SVM_classifier


sarcastic_loader = dataloader.DataLoader(img_dir="archive/Sarcasm_Headlines_Dataset_v2.json", train_size = 0.6)
svm_model = SVM_classifier.SVM_Text_Model(sarcastic_loader)
svm_model.load_model("filename.joblib")

print("evaluating...")
print(svm_model.evaluate())