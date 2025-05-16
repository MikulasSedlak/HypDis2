# Import necessary libraries
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report,  precision_score, recall_score, f1_score

import database as DB
import torch



def randForest(database):

    #creates matrixes
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


    splitCount = 0
    # Iterate over each fold
    #for train_idx, test_idx in kfold.split(X, y):
    for train_idx, test_idx in gkf.split(X, y, patientLabels):

        #split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = yDiagnosis[train_idx], yDiagnosis[test_idx]
        
        #random forest
        model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        model.fit(X_train, y_train)
        
        #predictions
        y_pred = model.predict(X_test)
        
        #add to the confusion matrix
        #database.confusionMatrix += confusion_matrix(y_test, y_pred)

        #save info back into the recording

        for idx, true_label, pred_label in zip(test_idx, y_test, y_pred):

            # Compute the confusion matrix for this single prediction
            single_conf_matrix = confusion_matrix([true_label], [pred_label], labels=[0, 1])
            # Save it to the corresponding recording
            database.recordings[idx].saveConfusionMatrix(single_conf_matrix)

        splitCount+=1
        
        if __debug__:
            print(f"Done {splitCount}/{k_splits}") 
    


    #save
    database.saveConfusionMatrixes()



#for loadpath in DB.loadpaths:
loadpath = "pyannote.csv"
#loads DB
database = DB.Database()
database.load(loadpath)

#load model
database.loadConfusionMatrixes()

#create model
#randForest(database)


#sorts into list of exercises
#DB.Database.Accuracy.mf(database)

#print exercises
#database.Accuracy.exercises(database)
#database.Accuracy.printspecialexcs(database, "7.")
#database.Accuracy.printspecialexcs(database, "8.")
#database.Accuracy.printspecialexcs(database, "9.")

#database.Accuracy.printMetrics(database)
#print("========================================================")

#DB.Database.Accuracy.sortExercises(database)

#makes list of only selected exercises [3]
exercises = [ex for ex in database.sortRecExerc() if ex[0].exerciseNumber in DB.selectedExercises]


        

        



