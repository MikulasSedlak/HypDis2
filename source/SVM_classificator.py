# Import necessary libraries
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report,  precision_score, recall_score, f1_score

import database as DB
import torch

def printspecialexcs(recordings,STR):
    cm = np.zeros((2, 2), dtype=int)
    for exercise in recordings:
        if STR in exercise.exerciseNumber:
            cm += exercise.confMatrix
    acu = (cm[1, 1]+cm[0,0])
    racy= (cm[1, 1]+cm[0,0]+cm[1, 0]+cm[0, 1])
    acuracy = acu/racy
    print("===================================================")
    print(STR," ex. confusion Matrix:")
    print(f"TP: {cm[1, 1]}", end="  ")
    print(f"FN: {cm[1, 0]}")
    print(f"FP: {cm[0, 1]}", end="  ")
    print(f"TN: {cm[0, 0]}")
    print(f"ex. accuracy is {acuracy:.2f} ({acu}/{racy})")


class Accuracy:
        
    @classmethod
    def exercises(self, database):
        recordingsSortEx = database.sortRecExerc()

        for exercise in recordingsSortEx:
            exConfMatrix = np.zeros((2, 2), dtype=int)
            accurate = 0
            for recording in exercise:
                exConfMatrix += recording.confMatrix
                if recording.accurate:
                    accurate+=1
            exAccuracyP = 100*accurate/len(exercise)
            print(f"Ex. {exercise[0].exerciseNumber} accuracy: {exAccuracyP:.2f} ({accurate}/{len(exercise)})")
            print(f"TP: {exConfMatrix[1, 1]}", end="  ")
            print(f"FN: {exConfMatrix[1, 0]}")
            print(f"FP: {exConfMatrix[0, 1]}", end="  ")
            print(f"TN: {exConfMatrix[0, 0]}")
            print("---------------------------------------------------")

    def mf(database):
        M,F = database.splitGender()
        mAcc, fAcc = 0,0
        mCM,fCM = np.zeros((2, 2), dtype=int),np.zeros((2, 2), dtype=int)
        for recording in M:
            mCM += recording.confMatrix
            if recording.accurate:
                mAcc+=1
        for recording in F:
            fCM += recording.confMatrix
            if recording.accurate:
                fAcc+=1
        mAccP = 100*mAcc/len(M)
        fAccP = 100*fAcc/len(F)
        print("---------------------------------------------------")
        print(f"Male accuracy is {mAccP:.2f} ({mAcc}/{len(M)})")
        print(f"Female accuracy is {fAccP:.2f} ({fAcc}/{len(F)})")
        print("---------------------------------------------------")
        print("Male CM:")
        print(f"TP: {mCM[1, 1]}", end="  ")
        print(f"FN: {mCM[1, 0]}")
        print(f"FP: {mCM[0, 1]}", end="  ")
        print(f"TN: {mCM[0, 0]}")
        print("---------------------------------------------------")
        print("Female CM:")
        print(f"TP: {fCM[1, 1]}", end="  ")
        print(f"FN: {fCM[1, 0]}")
        print(f"FP: {fCM[0, 1]}", end="  ")
        print(f"TN: {fCM[0, 0]}")
        print("---------------------------------------------------")

    def random(database):
        text = 0


loadpath = "pyannote.csv"
savepath = "MatrixTest.csv"

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

k_splits=3

#kfold = GroupKFold(n_splits, shuffle=True) #Shuffle = 
gkf = StratifiedGroupKFold(k_splits)


# Initialize metrics and results storage
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
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
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

totalAcuraccy = 100*(confusionMatrix[0, 0] + confusionMatrix[1, 1])/np.sum(confusionMatrix)

#save
DB.saveConfusionMatrix(savepath,database.recordings)

##accuracy of exercies

#sorts into list of exercises

Accuracy.exercises(database)
Accuracy.mf(database)

#print CM
print("---------------------------------------------------")
print("Confusion Matrix:")
print(f"TP: {confusionMatrix[1, 1]}", end="  ")
print(f"FN: {confusionMatrix[1, 0]}")
print(f"FP: {confusionMatrix[0, 1]}", end="  ")
print(f"TN: {confusionMatrix[0, 0]}")

# Calculate metrics
precision = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[0, 1])
recall = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)

# Print results
printspecialexcs(database.recordings, "7.")
printspecialexcs(database.recordings, "8.")
printspecialexcs(database.recordings, "9.")

print("===================================================")
print(f"Precision: {100*precision:.1f}")
print(f"Recall: {100*recall:.1f}")
print(f"F1 Score: {100*f1_score:.1f}")
print("===================================================")
