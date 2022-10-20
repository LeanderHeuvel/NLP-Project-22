import dataloader
from models import SVM_classifier
sarcastic_loader = dataloader.DataLoader(img_dir="archive/Sarcasm_Headlines_Dataset_v2.json", portion=0.2)
svm_model = SVM_classifier.SVM_Text_Model(sarcastic_loader)
print("training...")
svm_model.fit_model()
print("predicting...")
print(svm_model.predict(["dem rep. totally nails why congress is falling short on gender, racial equality",
                "thirtysomething scientists unveil doomsday clock of hair loss"]))