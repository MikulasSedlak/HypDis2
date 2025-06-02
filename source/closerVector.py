import database as DB
import torch
import math
import torch.nn.functional as F
import numpy as np

def absDist(vec):
    sum = 0
    for i in vec:
        sum += i*i
    return math.sqrt(sum)


def cosine_similarity(A, B):
        C = F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))
        return C.squeeze()




def sortRecExerc(recordings):
    recordingsSorted = []
    FileCount = 0
    for recording in recordings:
        recordingAdded = False
        if not recordingsSorted == []:
            for recGroup in recordingsSorted:
                if recGroup[0].exerciseNumber == recording.exerciseNumber:
                    recGroup.append(recording)
                    recordingAdded = True
                    break
        if not recordingAdded:
            recGroup = []
            recordingsSorted.append(recGroup)
            recordingsSorted[-1].append(recording)
        FileCount += 1
    return recordingsSorted


    
def splitPD(recordings): #takes group, returns two - P, K
    P = []
    K = []
    countP = countK = 0
    for recording in recordings:
        if recording.hasPD:
            P.append(recording)
            countP +=1
        else:
            K.append(recording)
            countK +=1
    return K,P

def splitG(recordings): #takes group, returns two - P, K
    M = []
    F = []
    countP = countK = 0
    for recording in recordings:
        if recording.isMale:
            M.append(recording)
            countP +=1
        else:
            F.append(recording)
            countK +=1
    return F,M



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


def weightedAvg(recordingsSorted):
    avgVec = []
    for exercise in recordingsSorted:

        #iterate through recordings in the exercise
        avgVec4Group = []
        for recording in exercise:
            # unsqueeze
            avgVec4Group.append(recording.audio2vec.unsqueeze(0))

        # Concatenate all the tensors in the list into a single tensor along dimension 0
        avgVec4Group = torch.cat(avgVec4Group, dim=0)

        #compute the median along the dimension 0
        avgVec4Group = torch.mean(avgVec4Group, dim=0) 

        #add to final
        avgVec.append(avgVec4Group.unsqueeze(0))

    # Concatenate all the tensors in the list into a single tensor along dimension 0
    avgVec = torch.cat(avgVec, dim=0)
    
    # Compute the median along the specified dimension (e.g., across rows or columns)
    avgVec = torch.mean(avgVec, dim=0)  # Use dim=0 or dim=1 based on the desired mean direction

    return avgVec

def weightedMed(recordingsSorted):
    medVec = []
    for exercise in recordingsSorted:
        
        medVec4Group = []

        #iterate through recordings in the exercise
        for recording in exercise:
            #ensure vector is unsqueezed and added to the list
            medVec4Group.append(recording.audio2vec.unsqueeze(0))

        # Concatenate all the tensors in the list into a single tensor along dimension 0
        medVec4Group = torch.cat(medVec4Group, dim=0)

        # Compute the median along the specified dimension (e.g., across rows or columns)
        medVec4Group, _ = torch.median(medVec4Group, dim=0)  # Use dim=0 or dim=1 based on the desired mean direction

        # Print size for debugging
        #print(medVec4group.size())

        # add to final
        medVec.append(medVec4Group.unsqueeze(0))
    # Concatenate all the tensors in the list into a single tensor along dimension 0
    medVec = torch.cat(medVec, dim=0)
    
    # Compute the median along the specified dimension (e.g., across rows or columns)
    medVec, _ = torch.median(medVec, dim=0)  # Use dim=0 or dim=1 based on the desired mean direction

    return medVec
    


def evaluateSimilarity(database, use_median=False, weighted=False):
    recordings = database.recordings

    if weighted:
        K, P = splitPD(recordings)
        Psort = sortRecExerc(P)
        Ksort = sortRecExerc(K)
        Psort = [sublist for sublist in Psort if len(sublist) > 10]
        Ksort = [sublist for sublist in Ksort if len(sublist) > 10]

        vecP = weightedMed(Psort) if use_median else weightedAvg(Psort)
        vecK = weightedMed(Ksort) if use_median else weightedAvg(Ksort)
    else:
        vecP, vecK = medianVec(recordings) if use_median else avgVec(recordings)

    method = "median" if use_median else "average"
    strategy = "weighted" if weighted else "simple"

    confMatrix = np.zeros((2, 2), dtype=int)

    for recording in recordings:
        simP = cosine_similarity(vecP, recording.audio2vec)
        simK = cosine_similarity(vecK, recording.audio2vec)

        true_label = 1 if recording.hasPD else 0
        pred_label = 1 if simP > simK else 0
        confMatrix[true_label, pred_label] += 1

    print(f"Confusion Matrix ({method}, {strategy}): {database.name}")
    print(f"TP: {confMatrix[1, 1]}", end="  ")
    print(f"FN: {confMatrix[1, 0]}")
    print(f"FP: {confMatrix[0, 1]}", end="  ")
    print(f"TN: {confMatrix[0, 0]}")

    matrix = DB.CM(confMatrix)
    precision = matrix.precision()
    recall = matrix.recall()
    f1_score = matrix.f1score()
    accuracy = matrix.accuracy()

    print(f"Accuracy: {100 * accuracy:.2f}")
    print(f"Precision: {100 * precision:.2f}")
    print(f"Recall: {100 * recall:.2f}")
    print(f"F1 Score: {100 * f1_score:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    databases = DB.loadAllDB()
    for database in databases:
        evaluateSimilarity(database, use_median=False, weighted=False)
        evaluateSimilarity(database, use_median=True, weighted=False)
        evaluateSimilarity(database, use_median=False, weighted=True)
        evaluateSimilarity(database, use_median=True, weighted=True)
