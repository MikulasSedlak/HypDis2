# Import necessary libraries
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report,  precision_score, recall_score, f1_score

import database as DB
import torch



def randForest(database, nEstimators=150, dummy = False): #nEstimators = number of trees, dummy = True switches Random forest to dummy classifier

    #creates matrixes
    X, y, patientLabels = database.toTensor()

    #shuffles before creating the folds
    torch.manual_seed(42)
    perm = torch.randperm(y.size(0)) #random permutation
    X = X[perm]
    y = y[perm]
    patientLabels = patientLabels[perm]

    #converts to numpy
    X = torch.Tensor.numpy(X)   #X features
    y = torch.Tensor.numpy(y)   #labels

    yDiagnosis = np.array([label % 10 for label in y])

    #kfold cross validation
    kSplits=10
    gkf = StratifiedGroupKFold(kSplits)
    splitCount = 0

    # Iterate over each fold
    for train_idx, test_idx in gkf.split(X, y, patientLabels):

        #split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = yDiagnosis[train_idx], yDiagnosis[test_idx]
        

        if dummy:
            dummy = DummyClassifier(strategy='most_frequent')
            dummy.fit(X_train, y_train)
            y_pred = dummy.predict(X_test)
        else:
            #random forest
            model = RandomForestClassifier(n_estimators=nEstimators, max_depth=None, random_state=42)
            model.fit(X_train, y_train)

            #probabilities for class 1
            probs = model.predict_proba(X_test)[:, 1]

            #decision threshold 
            threshold = 0.55
            y_pred = (probs > threshold).astype(int)

        #save CMs to recordings
        for idx, true_label, pred_label in zip(test_idx, y_test, y_pred):

            # Compute the confusion matrix for this single prediction
            singleConfMatrix = confusion_matrix([true_label], [pred_label], labels=[0, 1])
           
            # Save it to the corresponding recording
            database.recordings[idx].saveConfusionMatrix(singleConfMatrix)

        splitCount+=1
        
        if __debug__:
            print(f"k splits: {splitCount}/{kSplits}") 

    #database.saveConfusionMatrixes2()
    DB.Database.Accuracy.printMetrics(database)


def selectedExercises():
    #prints metrics for selected exercises
    databases = []
    for loadpath in DB.loadpaths:
        database = DB.Database()
        database.load(loadpath)
        databases.append(database)
        #load model
        database.loadConfusionMatrixes()

    F1ScoresAll = []

    for database in databases:
        #makes list of only selected exercises [3]
        exercises = [ex for ex in database.sortRecExerc() if ex[0].exerciseNumber in DB.selectedExercises]

        F1Scores = []
        for exercise in exercises:
            CMatrix = DB.CM()
            for recording in exercise:
                CMatrix += recording.confMatrix
            F1Scores.append(CMatrix.f1score())
        F1ScoresAll.append(F1Scores)

    #print f1 scores
    for i, column in enumerate(F1ScoresAll):
        print(DB.loadpaths[i], end="")
        for value in column:
            print(f" {100*value:.2f}", end ="")
            print()
    return F1ScoresAll
        


if __name__ == "__main__":
    
    databases = DB.loadAllDB()
    scoresAll = []

    for database in databases:
        scores = []
        CMatrix = DB.CM()
        for recording in database.recordings:
            CMatrix += recording.confMatrix
        scores.append(CMatrix.accuracy())
        scores.append(CMatrix.precision())
        scores.append(CMatrix.recall())
        scores.append(CMatrix.f1score())
        scoresAll.append(scores)
    #print f1 scores
    for i, column in enumerate(scoresAll):
        print(DB.loadpaths[i], end="")
        for value in column:
            print(f" {100*value:.2f}", end ="")
            print()

