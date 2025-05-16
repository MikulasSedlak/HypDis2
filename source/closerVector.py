import database as DB
import torch
import math
import torch.nn.functional as F

def absDist(vec):
    sum = 0
    for i in vec:
        sum += i*i
    return math.sqrt(sum)


def cosine_similarity(A, B):
        C = F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))
        return C.squeeze()

def avgVec(recordings):
    sumP = torch.empty(1)
    sumK = torch.empty(1)
    countP = countK = 0
    for recording in recordings:
        if recording.hasPD:
            sumP = torch.add(sumP, recording.audio2vec)
            countP +=1
        else:
            sumK = torch.add(sumK, recording.audio2vec)
            countK +=1
    avgP=torch.divide(sumP,countP)
    avgK=torch.divide(sumK,countK)
    print("K:", countK)
    print("P:", countP)
    #print("Average P vector: ", avgP)
    #print("Average K vector: ", avgK)
    return [avgP,avgK]

def medianVec(recordings):
    valuesP = []
    valuesK = []

    # Collect all audio2vec vectors
    for recording in recordings:
        if recording.hasPD:
            valuesP.append(recording.audio2vec)
        else:
            valuesK.append(recording.audio2vec)


    # Calculate medians
    medP = torch.median(torch.stack(valuesP), dim=0).values if valuesP else None
    medK = torch.median(torch.stack(valuesK), dim=0).values if valuesK else None
    return medP, medK

def calculateSimilarity(database):
    avgP, avgK = avgVec(database.recordings)
    medP, medK = medianVec(database.recordings)

    TP = 0
    FP=0
    FN = 0
    TN = 0


    for recording in database.recordings:
        sim_k = cosine_similarity(avgK, recording.audio2vec)  # Similarity to avgK
        sim_p = cosine_similarity(avgP, recording.audio2vec)  # Similarity to avgP
        
        # Compare similarities: closer to avgP or avgK?
        if sim_p > sim_k:  # Closer to avgP (Parkinson's)
            if recording.hasPD:
                TP += 1  # Correctly predicted Parkinson's
            else:
                FP += 1  # Incorrectly predicted Parkinson's
        else:  # Closer to avgK (Non-Parkinson's)
            if recording.hasPD:
                FN += 1  # Incorrectly predicted Non-Parkinson's
            else:
                TN += 1  # Correctly predicted Non-Parkinson's

    
    # Print confusion matrix
    print("Confusion Matrix avg:")
    print(f"TP: {TP}, FN: {FN}")
    print(f"FP: {FP}, TN: {TN}")
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN) 
    f1_score = 2 * precision * recall / (precision + recall)
    acuracy = (TP + TN)/len(database.recordings)
    print(f"Acurracy: {100*acuracy:.2f}")
    print(f"Precision: {100*precision:.2f}")
    print(f"Recall: {100*recall:.2f}")
    print(f"F1 Score: {100*f1_score:.2f}")

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for recording in database.recordings:
        sim_k = cosine_similarity(medK, recording.audio2vec)  # Similarity to avgK
        sim_p = cosine_similarity(medP, recording.audio2vec)  # Similarity to avgP
        
        # Compare similarities: closer to avgP or avgK?
        if sim_p > sim_k:  # Closer to avgP (Parkinson's)
            if recording.hasPD:
                TP += 1  # Correctly predicted Parkinson's
            else:
                FP += 1  # Incorrectly predicted Parkinson's
        else:  # Closer to avgK (Non-Parkinson's)
            if recording.hasPD:
                FN += 1  # Incorrectly predicted Non-Parkinson's
            else:
                TN += 1  # Correctly predicted Non-Parkinson's

    #DB.saveDistancefKP("distancesAud2vec7.csv",recordings)
    # Print confusion matrix

    print("__________________________________________________")
    print("Confusion Matrix MED:")
    print(f"TP: {TP}, FN: {FN}")
    print(f"FP: {FP}, TN: {TN}")
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN) 
    f1_score = 2 * precision * recall / (precision + recall)
    acuracy = (TP + TN)/len(database.recordings)
    print(f"Acurracy: {100*acuracy:.2f}")
    print(f"Precision: {100*precision:.2f}")
    print(f"Recall: {100*recall:.2f}")
    print(f"F1 Score: {100*f1_score:.2f}")
    print("===================================================")



#loads DB
for loadpath in DB.loadpaths:
    print(loadpath)
    database = DB.Database()
    database.load(loadpath)
    calculateSimilarity(database)
ç