import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import database as DB
import os

def getPDvecsIF(recordings,exerciseSTR): #gets ifvecs ...
    isExvec=[]
    noExvec=[]
    for recording in recordings: 
        if exerciseSTR in recording.exerciseNumber: #IF
            isExvec.append(recording.audio2vec)
        else:
            noExvec.append(recording.audio2vec)
    return isExvec, noExvec

def getPDvecsSame(recordings,exerciseSTR): #gets ifvecs ...
    isExvec=[]
    noExvec=[]
    for recording in recordings: 
        if recording.patientNumber == "P1027": #IF
            isExvec.append(recording.audio2vec)
        else:
            noExvec.append(recording.audio2vec)
    return isExvec, noExvec
def getPDmalefemale(recordings): #splits into male female.
    isMale=[]
    isFemale=[]
    for recording in recordings: 
        if recording.isMale: #IF
            isMale.append(recording.audio2vec)
        else:
            isFemale.append(recording.audio2vec)
            
    return isMale, isFemale

def getPDvecsjustIF(recordings,exerciseSTR): #gets PDvecs just if ...
    hasPD=[]
    noPD=[]
    for recording in recordings: 
        if exerciseSTR in recording.exerciseNumber: #IF
            if recording.hasPD: #IF
                hasPD.append(recording.audio2vec)
            else:
                noPD.append(recording.audio2vec)
    return hasPD, noPD

def graphIF(database, gender = False, ex = "7."): 
    #graphs, based on a condition 
    """gender=True to print gender differences
    ex="Exercise number", to print avilaible different numbers run DB.Database.Accuracy.sortExercises(database)"""

    #aquire data
    if gender:
        Group1, Group2 = getPDmalefemale(database.recordings)
        G1Name = "muži"
        G2Name = "ženy"
        print(f"creating tSNE - {database.name}, gendered")
    else:
        Group1, Group2 = getPDvecsIF(database.recordings,ex)
        G1Name = "fonace"
        G2Name = "ostatní"
        print(f"creating tSNE - {database.name}, exercise {ex}")

    Group1 = np.array(Group1)
    Group2 = np.array(Group2)

    data = np.concatenate([Group1, Group2])

    #apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(Group1), 0], reduced_data[:len(Group1), 1], s=1, color='red', alpha=0.7, label=G1Name)
    plt.scatter(reduced_data[len(Group1):, 0], reduced_data[len(Group1):, 1], s=1, color='blue', alpha=0.7, label=G2Name)
    if gender:
        plt.title("t-SNE " + database.name + ", dle pohlaví")
        typeName = "_malefemale"
    else:
         plt.title("t-SNE " + database.name + ", " + G1Name)
         typeName = "_" + G1Name
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(False)

    #hide axis numbers
    plt.gca().set_xticks([])  
    plt.gca().set_yticks([]) 

    #save picture
    dir = "tsne/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + database.name + typeName + '.png', dpi=300)
    #plt.show()

def graphBasic(database):
    print(f"creating tSNE - {database.name}, PD")
    #aquire data
    hasPDvec, noPDvec = database.getPDvecs()
    hasPDvec = np.array(hasPDvec)
    noPDvec = np.array(noPDvec)
    data = np.concatenate([hasPDvec, noPDvec])

    #apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    #plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(hasPDvec), 0], reduced_data[:len(hasPDvec), 1], s=1, color='red', alpha=0.7, label='HD')
    plt.scatter(reduced_data[len(hasPDvec):, 0], reduced_data[len(hasPDvec):, 1], s=1, color='blue', alpha=0.7, label='bez HD')
    plt.title("t-SNE " + database.name)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(False)

    #hide axis numbers
    plt.gca().set_xticks([])  
    plt.gca().set_yticks([]) 

    #save picture
    dir = "tsne/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir + database.name + '.png', dpi=300)
    #plt.show()

if __name__ == "__main__":
    databases = DB.loadAllDB()
    for database in databases: 
        graphBasic(database)
        graphIF(database, gender=False)
        graphIF(database, gender=True)
