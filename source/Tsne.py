import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import database as DB

def graph1(recordings, filename):
    hasPDvec, noPDvec = DB.getPDvecs(recordings)
    
    hasPDvec = np.array(hasPDvec)
    noPDvec = np.array(noPDvec)

    data = np.concatenate([hasPDvec, noPDvec])

    #tSNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    #plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(hasPDvec), 0], reduced_data[:len(hasPDvec), 1], s=1, color='red', alpha=0.7, label='HD')
    plt.scatter(reduced_data[len(hasPDvec):, 0], reduced_data[len(hasPDvec):, 1], s=1, color='blue', alpha=0.7, label='bez HD')
    title = "t-SNE " + "Audio2vec"
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.gca().set_xticks([])  # Hide x-axis numbers
    plt.gca().set_yticks([]) 
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.savefig('Audio2vec.png', dpi=300)
    plt.savefig('Audio2vec.svg', bbox_inches='tight', transparent=True)


def graph3():
    recordings = DB.loadDB("audio2vec100.csv")
    hasPDvec, noPDvec = DB.getPDvecs(recordings)
    hasPDvec = np.array(hasPDvec)
    noPDvec = np.array(noPDvec)

    data = np.concatenate([hasPDvec, noPDvec])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(hasPDvec), 0], reduced_data[:len(hasPDvec), 1], s=1, color='red', alpha=0.7, label='HD')
    plt.scatter(reduced_data[len(hasPDvec):, 0], reduced_data[len(hasPDvec):, 1], s=1, color='blue', alpha=0.7, label='bez HD')
    plt.title("t-SNE Vizualizace Audio2vec(7)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def graph4(recordings): #wav2vec just phonation exercises
    
    hasPDvec, noPDvec = getPDvecsjustIF(recordings, "8.")
    hasPDvec = np.array(hasPDvec)
    noPDvec = np.array(noPDvec)

    data = np.concatenate([hasPDvec, noPDvec])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(hasPDvec), 0], reduced_data[:len(hasPDvec), 1], s=10, color='red', alpha=0.7, label='HD')
    plt.scatter(reduced_data[len(hasPDvec):, 0], reduced_data[len(hasPDvec):, 1], s=10, color='blue', alpha=0.7, label='bez HD')
    plt.title("t-SNE Open3L - jen intonace")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.gca().set_xticks([])  # Hide x-axis numbers
    plt.gca().set_yticks([]) 
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.savefig('OpenL3jInt.png', dpi=300)
    plt.savefig('OpenL3jInt.svg', bbox_inches='tight', transparent=True)
    plt.show()
    

def graph5(recordings): #phonation
    
    isExvec, noExvec = getPDvecsIF(recordings, "9.")
    isExvec = np.array(isExvec)
    noExvec = np.array(noExvec)

    data = np.concatenate([isExvec, noExvec])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(isExvec), 0], reduced_data[:len(isExvec), 1], s=2, color='blue', alpha=0.7, label='fonace')
    plt.scatter(reduced_data[len(isExvec):, 0], reduced_data[len(isExvec):, 1], s=1, color='red', alpha=0.7, label='ostatní')
    plt.title("t-SNE Open3L - rychle opakovaná slova")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.gca().set_xticks([])  # Hide x-axis numbers
    plt.gca().set_yticks([]) 
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.savefig('Open3LRych.png', dpi=300)
    plt.savefig('Open3LRych.svg', bbox_inches='tight', transparent=True)

def graphpatient(recordings): #phonation
    
    isExvec, noExvec = getPDvecsSame(recordings, "10.4-2")
    isExvec = np.array(isExvec)
    noExvec = np.array(noExvec)

    data = np.concatenate([isExvec, noExvec])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(isExvec), 0], reduced_data[:len(isExvec), 1], s=1, color='blue', alpha=0.7, label='cvič. 0.0')
    plt.scatter(reduced_data[len(isExvec):, 0], reduced_data[len(isExvec):, 1], s=1, color='red', alpha=0.7, label='ostatní')
    plt.title("t-SNE Audio2vec - cvičení 0.0")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(True)
    plt.show()

def graph7(): #wav2vec just chosen words
    recordings = DB.loadDB("wav2vec.csv")
    hasPDvec, noPDvec = getPDvecsjustIF(recordings, "10")
    hasPDvec = np.array(hasPDvec)
    noPDvec = np.array(noPDvec)

    data = np.concatenate([hasPDvec, noPDvec])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(hasPDvec), 0], reduced_data[:len(hasPDvec), 1], s=5, color='blue', alpha=0.7, label='HD')
    plt.scatter(reduced_data[len(hasPDvec):, 0], reduced_data[len(hasPDvec):, 1], s=5, color='red', alpha=0.7, label='bez HD')
    plt.title("t-SNE Vizualizace Wav2vec, jen fonace")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(True)
    plt.show()

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

def graph6(recordings): #gender
    isMale, isFemale = getPDmalefemale(recordings)
    isMale = np.array(isMale)
    isFemale = np.array(isFemale)

    data = np.concatenate([isMale, isFemale])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(isMale), 0], reduced_data[:len(isMale), 1], s=1, color='blue', alpha=0.7, label='muži')
    plt.scatter(reduced_data[len(isMale):, 0], reduced_data[len(isMale):, 1], s=1, color='red', alpha=0.7, label='ženy')
    plt.title("t-SNE OpenL3 - pohlaví")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.gca().set_xticks([])  # Hide x-axis numbers
    plt.gca().set_yticks([]) 
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.subplots_adjust(bottom=0.07, top=0.93)
    plt.savefig('OpenL3GEN.png', dpi=300)
    plt.savefig('OpenL3GEN.svg', bbox_inches='tight', transparent=True)
    plt.show()

def graphBasic(database):
    hasPDvec, noPDvec = DB.getPDvecs(database.recordings)
    hasPDvec = np.array(hasPDvec)
    noPDvec = np.array(noPDvec)

    data = np.concatenate([hasPDvec, noPDvec])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    reduced_data = tsne.fit_transform(data)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:len(hasPDvec), 0], reduced_data[:len(hasPDvec), 1], s=1, color='red', alpha=0.7, label='HD')
    plt.scatter(reduced_data[len(hasPDvec):, 0], reduced_data[len(hasPDvec):, 1], s=1, color='blue', alpha=0.7, label='bez HD')
    plt.title("t-SNE " + database.name)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.grid(True)
    plt.show()


filename = "VGGish.csv"
data = DB.Database()
data.load(filename)
graphBasic(data)

