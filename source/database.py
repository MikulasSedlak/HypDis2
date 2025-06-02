import os
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import resampy
import rForest as rF
import torch
from scipy.sparse import coo_matrix
from audio2vec import Audio2Vec
import pyannote.audio
import wave
import librosa
import numpy as np
#splitters hierarchy in ".csv"
SP1 = ","
SP2 = "/"
SP3 = "_"

loadpaths = ["wav2vec.csv", "audio2vec512.csv", "openL3-mean.csv", "pyannote.csv", "VGGish.csv"]
selectedExercises = ["7.1-1-e","7.4-3","9.4"]


class CM: #confusion matrix
    def __init__(self, matrix = [[0,0],[0,0]]):
        self.matrix = self.formatMatrix(matrix)
    
    def __call__(self):
        return self.matrix
    
    def __add__(self, other):
        newMatrix = np.array(self.matrix) + np.array(other)
        return CM(newMatrix)
    
    def precision(self):
        TN, FP = self.matrix[0]
        FN, TP = self.matrix[1]
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

    def accuracy(self):
        TN, FP = self.matrix[0]
        FN, TP = self.matrix[1]
        if (TP + FN + FP + TN) == 0:
            return 0.0
        return (TN + TP)/ (TP + FN + FP + TN)
    
    def recall(self):
        TN, FP = self.matrix[0]
        FN, TP = self.matrix[1]
        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    def f1score(self):
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def formatMatrix(self,matrix):
        #convert numpy arrays to list
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
        #convert torch tensors to list
        else: 
            if isinstance(matrix, torch.Tensor):
                matrix = matrix.tolist()
        #raise error
        if not isinstance(matrix, list):
            raise TypeError("Matrix must be a list, numpy.ndarray, or torch.Tensor.")
        #shape
        if len(matrix) != 2:
            raise ValueError("Matrix must have exactly 2 rows.")
        #correct type
        for i, row in enumerate(matrix):
            if not isinstance(row, list):
                raise TypeError(f"Row {i} is not a list.")
            if len(row) != 2:
                raise ValueError(f"Row {i} must have exactly 2 elements.")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Element at position [{i}][{j}] must be an int or float.")
        
        return matrix 
    
    def isAccurate(self):
        if self.matrix[0][0] == 1 or self.matrix[1][1] == 1:
            return True
        return False
    
            

class Recording:
    def __init__(self,recording): #initialization is one line from files.csv, can be also initialized form other ".csv" databases
            #contains booleans: hasPD, isMale, 
        
        recording = recording.split(SP1) 
        name = recording[0] 
        self.path = str(name)
        name = name.split(SP2)
        name = name[-1] #just the name of recording      
        name = name.split(SP3)
  
        if name == "1.wav":
            name.pop()
        #PD diagnosis
        if name[0][0]=="P":
            self.hasPD = True
        else:
            self.hasPD = False

        #patient number
        self.patientNumber = name[0]

        #exercise id
        self.exerciseNumber = name[1]

        #gender
        if name[0][1] == "2":
            self.isMale = True
        else: 
            self.isMale = False

        #checks if it contains vectors
        if(len(recording)>1):
            self.audio2vecStr = recording[1]
            self.audio2vec = self.audio2vecStr.split(SP2)
            self.audio2vec = [float(cell) for cell in self.audio2vec]
            self.audio2vec = torch.tensor(self.audio2vec)
        else:
            self.audio2vecStr =""

        self.distanceFromKavg,self.distanceFromPavg,self.distanceFromKmed,self.distanceFromPmed = 0,0,0,0

    def audio2vecSave(self,audiovector):
        if not isinstance(audiovector, list):
            audiovector = audiovector.tolist()
        self.audio2vec = audiovector
        self.audio2vecStr = SP2.join(map(str,audiovector))

    #saves confusion matrix to confMatrix
    def saveConfusionMatrix(self,singleConfMatrix):
        self.confMatrix = singleConfMatrix
        if singleConfMatrix[0,0] == 1 or singleConfMatrix[1,1] == 1:
            self.accurate = True
        else:
            self.accurate = False
    



def get_all_file_paths(dir="resources/recordings"):   
    file_paths = []
    filecount = 0
    for root, _, files in os.walk(dir):
        for file in files:
            if file[0]==".":
               continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
            filecount+=1
            #print("filecount: ", filecount)
    return file_paths 


class Patient:
    def __init__(self, recording):
        self.hasPD = recording.hasPD
        self.isMale = recording.isMale
        self.number = recording.patientNumber
        self.recordingsList = []
        self.recordingsList.append(recording)



class Database:

    #summing confusion matrix
    def confMatrix(self):
        confMatrix = np.array([[0,0],[0,0]])
        for recording in self.recordings:
            confMatrix+=recording.confMatrix
        return confMatrix

    #copies database
    def make(self,recordings):
        self.recordings = recordings
        self.patients = self.loadPatients()

    #loads database from file
    def load(self,databaseFile="files.csv"):
        self.databaseFilename = databaseFile
        name = databaseFile.split(".")
        self.name = name[0]

        if __debug__:
            print("loading ", databaseFile, "...")
        databasePath ="resources/databases/" + databaseFile
        recordings = []
        with open(databasePath, "r") as openfileobject:
            for line in openfileobject:
                recordings.append(Recording(line.strip()))
        if __debug__:
            print(databasePath," loaded.")

        self.recordings = recordings
        self.patients = self.loadPatients()
    
    #returns list of patients with their recordings
    def loadPatients(self):
        patientList = []
        
        for recording in self.recordings:
            if  any(patient.number == recording.patientNumber for patient in patientList):
                patientList[-1].recordingsList.append(recording)
            else:
                patientList.append(Patient(recording))
                patientsCount = len(patientList)
        return patientList
    

    #saves embeddings to a file (fileObjects = name_of_a_file.csv)
    def save(self, fileObjects=None):
        if fileObjects == "files.csv":
            raise NameError("Cannot overwrite files.csv")           
        if fileObjects is None:
            fileObjects = self.databaseFilename
        file ="resources/databases/" + fileObjects
        f = open(file, "w")
        for recording in self.recordings:
            f.writelines([recording.path,SP1,recording.audio2vecStr,os.linesep])
        f.close()
        print("Database has been saved to file", fileObjects)

     #saves CMs to a file
    def saveConfusionMatrixes(self,file=None):
        if file is None:
            file ="resources/confusionMatrixes/" + self.name +"_CM" + ".csv"
        f = open(file, "w")
        
        for recording in self.recordings:
            if recording.accurate is None:
                raise ValueError("Conf. Matrix cannot be saved.")
            matrixStr = SP2.join(map(str,[item for sublist in recording.confMatrix for item in sublist]))
            f.writelines([str(recording.path),SP1,matrixStr,os.linesep])

        f.close()

    #loads CMs from a file
    def loadConfusionMatrixes(self,filename=None):
        if filename is None:
            filename = self.name + "_CM" + ".csv"

        #if file does not exist
        if not os.path.isfile(filename):
            return False
        print("loading ", filename, "...")
        databasePath ="resources/ConfusionMatrixes/" + filename
        with open(databasePath, "r") as openfileobject:
            for line in openfileobject:
                
                #splits into path
                linestr = line.strip()
                linestr= linestr.split(SP1)
                idx = next((i for i, recording in enumerate(self.recordings) if recording.path == linestr[0]), ValueError(f"Recording with path {linestr[0]} not found."))

                #create and save Conf matrix
                values = linestr[1].split(SP2)
                newMatrix = np.array([[int(values[0]), int(values[1])], [int(values[2]), int(values[3])]])
                self.recordings[idx].saveConfusionMatrix(newMatrix)
        
        
        print(f"{filename} succesfully made.")
        return True

    #returns two lists of recordings: hasPDvec, noPDvec
    def getPDvecs(self):
        hasPDvec=[]
        noPDvec=[]
        for recording in self.recordings:
            if recording.hasPD:
                hasPDvec.append(recording.audio2vec)
            else:
                noPDvec.append(recording.audio2vec)
        return hasPDvec, noPDvec
    
    #makes audio2Vec embeddings from recordings
    def makeAudio2Vec(self, n,recordings):
        processor = Audio2Vec(n)
        fileCount = 0
        for recording in recordings:
            filepath = recording.path
            # checking if it is a file
            if os.path.isfile(filepath):

                data = processor.audio2VectorProcessor(filepath)
                dense_tensor = torch.zeros((1, n), dtype=torch.float32)
                
                coo_data = data.tocoo()
                rows = coo_data.row
                cols = coo_data.col
                values = coo_data.data
                print(len(values))

                # Fill the dense tensor using extracted data
                for row, col, value in zip(rows, cols, values):
                    dense_tensor[row, col] = value
                dense_tensor = torch.squeeze(dense_tensor)
                recording.audio2vecSave(dense_tensor)
                
                fileCount+=1
                print("filecount: ",fileCount)
        print(f"All embeddings of {n} features, were created !")
        self.recordings = recordings    

    #makes pyannote embeddings from recordings
    def makePyannote(self):
        #load pre-trained speaker embedding model
        #must have your authorisation token from huggingface.com
        use_auth_token = "REPLACE_WITH_YOUR_TOKEN"
        if use_auth_token == "REPLACE_WITH_YOUR_TOKEN":
            raise ValueError("Replace use_auth_token with a valid token form huggingface.com")
        model = Model.from_pretrained("pyannote/embedding", use_auth_token, revision="main", strict=False)
        inference = Inference(model, window="sliding")
        from pyannote.audio import Model
        from pyannote.audio import Inference
        for recording in self.recordings:
            filepath = recording.path
            # checking if it is a file
            if os.path.isfile(filepath):
                embedding = inference(filepath)
                embedding = np.median(embedding.data, axis=0)
                recording.audio2vecSave(embedding)

    #makes VGGish embeddings from recordings
    def makeVGGish(self):
        def load_audio(file_path, target_sr=16000):
            audio, sr = sf.read(file_path)
            if sr != target_sr:
                audio = resampy.resample(audio, sr, target_sr)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  #convert to mono
            import numpy as np

            #minimum length is 1 s
            if len(audio) < target_sr:
                pad_width = target_sr - len(audio)
                audio = np.pad(audio, (0, pad_width), mode='constant')
            return audio

        # Load VGGish model from TF Hub
        model = hub.load('https://tfhub.dev/google/vggish/1')
        count = 0
        # Process audio and get embeddings
        for recording in self.recordings:
            audio = load_audio(recording.path)
            audioTensor = tf.convert_to_tensor(audio, dtype=tf.float32)

            # Add batch dimension and get embeddings
            
            embedding = model(audioTensor)


            # Check if there are NaN or Inf values in the audio
            
            #print(f"COUNT:{count}")
            #print("Embedding shape:", embedding.shape)
            embedding = tf.reduce_mean(embedding, axis=0) #pooling
            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                raise TypeError("Embedding contains NaNs or Infs!")
                #print("Embedding shape:", embedding.shape)
            embedding = embedding.numpy().tolist()
            recording.audio2vecSave(embedding)
            #print("")
            #count+=1

    #returns matrix with embeddings, vector of labels (1 for has PD, 1 for CG) and patientLabels (same number for all patient's recordings)
    def toTensor(self, labelsON=True): 
        
        matrix, labels, patientLabels = [], [], []
        personsUsed = []
        
        for recording in self.recordings:
            matrix.append(recording.audio2vec)
            #add labels
            ylabel = 0
            if recording.hasPD:
                ylabel=1
            if recording.isMale:
                ylabel+=10
            labels.append(ylabel)

            #add personLabels
            if not recording.patientNumber in personsUsed:
                personsUsed.append(recording.patientNumber)
            patientLabels.append(personsUsed.index(recording.patientNumber))
            
        matrix = torch.stack(matrix)
        labels = torch.tensor(labels)
        patientLabels = torch.tensor(patientLabels)
        if labelsON:
            return matrix, labels, patientLabels
        
        else:
            return matrix,

    #returns list of exercises 
    def sortRecExerc(self):
        recordingsSorted = []
        #FileCount = 0
        for recording in self.recordings:
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
            #FileCount += 1
            #print("File: ", FileCount, "/",len(recordings)) #checking print
        return recordingsSorted

    #returns list of patients
    def sortRecPatients(self):
        recordingsSorted = []
        #FileCount = 0
        for recording in self.recordings:
            recordingAdded = False
            if not recordingsSorted == []:
                for recGroup in recordingsSorted:
                    if recGroup[0].patientNumber == recording.patientNumber:
                        recGroup.append(recording)
                        recordingAdded = True
                        break
            if not recordingAdded:
                recGroup = []
                recordingsSorted.append(recGroup)
                recordingsSorted[-1].append(recording)
            #FileCount += 1
            #print("File: ", FileCount, "/",len(recordings)) #checking print
        return recordingsSorted
    
    #returns two lists of recordings
    def splitGender(self): #takes group, returns two - M,F
        M = []
        F = []
        countM = countF = 0
        for recording in self.recordings:
            if recording.isMale:
                M.append(recording)
                countM +=1
            else:
                F.append(recording)
                countF +=1
        return M,F
    
    #class for printing statistical values
    class Accuracy:

        #calculates and prints CM of each exercise
        def exercises(database):
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
                print("-" * 50)

        
        #sorts and prints exercises accroding their number of recordings
        def sortExercises(database):
            recordingsSortEx = database.sortRecExerc()
            pairs = []
            for exercise in recordingsSortEx:
                pair = exercise[0].exerciseNumber,len(exercise)
                pairs.append(pair)
            sortedPairs = sorted(pairs, key=lambda x: x[0], reverse=True)

            print("list of exercises:")
            for number,times in sortedPairs:
                print(f"{number}: {times}")
            print("-" * 50)

        #deletes exercises that have less than minimum of recordings
        def excludeExercises(database, minimum= 90):
            recordingsSortEx = database.sortRecExerc()
            excludePairs = []
            deleted = 0
            for exercise in recordingsSortEx:
                pair = exercise[0].exerciseNumber,len(exercise)
                if pair[1]<minimum:
                    excludePairs.append(pair[0])
            for recording in database.recordings:
                if recording.exerciseNumber in excludePairs:
                    database.recordings.remove(recording)
                    deleted += 1
            
            print("recordings deleted = ", deleted)
            print("-" * 50)

        #calculates and prints metrics for each gender 
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
            print("-" * 50)
            print(f"Male accuracy is {mAccP:.2f} ({mAcc}/{len(M)})")
            print(f"Female accuracy is {fAccP:.2f} ({fAcc}/{len(F)})")
            print("-" * 50)
            print("Male CM:")
            print(f"TP: {mCM[1, 1]}", end="  ")
            print(f"FN: {mCM[1, 0]}")
            print(f"FP: {mCM[0, 1]}", end="  ")
            print(f"TN: {mCM[0, 0]}")
            print("-" * 50)
            print("Female CM:")
            print(f"TP: {fCM[1, 1]}", end="  ")
            print(f"FN: {fCM[1, 0]}")
            print(f"FP: {fCM[0, 1]}", end="  ")
            print(f"TN: {fCM[0, 0]}")
            print("-" * 50)

        #calculates and prints metrics for specific exercise
        def printspecialexcs(database,STR):
            cm = np.zeros((2, 2), dtype=int)
            for exercise in database.recordings:
                if STR in exercise.exerciseNumber:
                    cm += exercise.confMatrix
            acu = (cm[1, 1]+cm[0,0])
            racy= (cm[1, 1]+cm[0,0]+cm[1, 0]+cm[0, 1])
            acuracy = acu/racy
            print("-" * 50)
            print(STR," ex. confusion Matrix:")
            print(f"TP: {cm[1, 1]}", end="  ")
            print(f"FN: {cm[1, 0]}")
            print(f"FP: {cm[0, 1]}", end="  ")
            print(f"TN: {cm[0, 0]}")
            print(f"ex. accuracy is {acuracy:.2f} ({acu}/{racy})")

        #calculates and prints metrics for whole database
        def printMetrics(database):
            confusionMatrix = database.confMatrix()
            print("-" * 50)
            print("Confusion Matrix:")
            print(f"TP: {confusionMatrix[1, 1]}", end="  ")
            print(f"FN: {confusionMatrix[1, 0]}")
            print(f"FP: {confusionMatrix[0, 1]}", end="  ")
            print(f"TN: {confusionMatrix[0, 0]}")

            acuracy =  (confusionMatrix[1,1] + confusionMatrix[0, 0])/(confusionMatrix[1,1] + confusionMatrix[0, 1]+confusionMatrix[1,0] + confusionMatrix[0,0])
            precision = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[0, 1])
            recall = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[1, 0])
            f1_scored = 2 * (precision * recall) / (precision + recall)

            print("-" * 50)
            print("TOTAL")
            print(f"Acurracy: {100*acuracy:.2f}")
            print(f"Precision: {100*precision:.2f}")
            print(f"Recall: {100*recall:.2f}")
            print(f"F1 Score: {100*f1_scored:.2f}")
            print("-" * 50)


            
#number of samples
def getSampleNumber(filepath):
        with wave.open(filepath, "r") as wf:
            num_samples = wf.getnframes()
        return num_samples

#if a recording is too short fixes its size
def temp_fixWholeWindow(recordings):
    # Checks if the recording isnt too short (Since some recordings had a feature extracting problem when file was under 1 sec) 
    import soundfile as sf
    # Load the audio
    y, sr = librosa.load(filepath, sr=None)
    duration = len(y) / sr  # Calculate duration in seconds
    print(f"Original Duration: {duration} seconds")

    # Ensure at least 1 second of audio
    min_duration = 0.25  # Minimum required duration in seconds
    if duration < min_duration:
        padding = int(sr * min_duration) - len(y)  # Calculate padding length
        y = np.pad(y, (0, padding), mode="constant")  # Pad with zeros
        print(f"Padded Duration: {len(y) / sr} seconds!.")

        # Save the padded file temporarily
        temp_filepath = filepath.replace(".wav", "_padded.wav")
        sf.write(temp_filepath, y, sr)
        filepath = temp_filepath  # Update filepath to padded file

#returns a list of all loaded databases 
def loadAllDB():

    databases = []
    for loadpath in loadpaths:
        database = Database()
        database.load(loadpath)

        #load model
        if not database.loadConfusionMatrixes():
            #if it doesnt exist it runs rForest
            print("Confusion matrixes do not exist, running random forest")
            rF.randForest(database)
            database.saveConfusionMatrixes()
        databases.append(database)
    
    print("All databases loaded.")
    return databases
