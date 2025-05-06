import database as DB
import torch
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix

#filenames
loadpath = "pyannote.csv"
savepath = "VGGish_CM_SVM.csv"

#loads DB
database = DB.Database()
database.load(loadpath)


#creates matrixes,

X, y, patientLabels = database.toTensor()

#maybe unnecessary, shuffles before creating the folds
torch.manual_seed(42)
perm = torch.randperm(y.size(0)) #random permutation
X = X[perm]
y = y[perm]
patientLabels = patientLabels[perm]

#features
X = torch.Tensor.numpy(X)
#labels
y = torch.Tensor.numpy(y)

yDiagnosis = np.array([label % 10 for label in y])

#kfold Cross Validation

k_splits=10

#kfold = GroupKFold(n_splits, shuffle=True) #Shuffle = 
gkf = StratifiedGroupKFold(k_splits)


#initialize and results storage
classification_reports = []
confusionMatrix = np.zeros((2, 2), dtype=int)
splitCount = 0
# Iterate over each fold
#for train_idx, test_idx in kfold.split(X, y):
for train_idx, test_idx in gkf.split(X, y, patientLabels):

    #split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = yDiagnosis[train_idx], yDiagnosis[test_idx]
    
    #SVM training
    #model = LinearSVC(C=10.0, max_iter=1000) # Example parameters
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train, y_train)
    
    #predictions
    y_pred = model.predict(X_test)
    
    #add to the confusion matrix
    confusionMatrix += confusion_matrix(y_test, y_pred)
 
    #save info back into the recording

    for idx, true_label, pred_label in zip(test_idx, y_test, y_pred):

        # Compute the confusion matrix for this single prediction
        single_conf_matrix = confusion_matrix([true_label], [pred_label], labels=[1, 0])
        
        # Save it to the corresponding recording
        database.recordings[idx].saveConfusionMatrix(single_conf_matrix)

    #store classification report
    #classification_reports.append(classification_report(y_test, y_pred, output_dict=True))
    splitCount+=1
    print(f"Done {splitCount}/{k_splits}")

    #sorts into list of exercises

DB.Database.Accuracy.exercises(database)
DB.Database.Accuracy.mf(database)

#print CM
print("---------------------------------------------------")
print("Confusion Matrix:")
print(f"TP: {confusionMatrix[1, 1]}", end="  ")
print(f"FN: {confusionMatrix[1, 0]}")
print(f"FP: {confusionMatrix[0, 1]}", end="  ")
print(f"TN: {confusionMatrix[0, 0]}")
