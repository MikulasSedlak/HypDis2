import os
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import resampy
import pyannote.audio
import pyannote.pipeline
import torch
import torchaudio
import math
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
testfilename = "resources/recordings/K/K1003/K1003_0.0_1.wav"

loadpaths = ["wav2vec.csv", "audio2vec512.csv", "openL3-mean.csv", "pyannote.csv", "VGGish.csv"]
selectedExercises = ["7.1-1-e","7.4-3","9.4"]

##Yet unused
#def get_unique_filename(filename):
#        base, ext = os.path.splitext(filename)
#        counter = 1
#        while os.path.exists(filename):
#            filename = f"{base}_{counter}{ext}"
#            counter += 1
#        return filename

#TODO class CM, that calculates Precison, Recall, F1 score...

class CM:
    def __init__(self, matrix):
        if not self.correctSize(matrix):
            raise TypeError("Matrix size must be 2x2!")
        self.matrix = self.formatMatrix(matrix)
        return self.matrix
    
    def __call__(self):
        return self.matrix
    
    def __add__(self, other):
        newMatrix = np.array(self.matrix) + np.array(other.matrix)
        return CM(newMatrix)
    
    def precision(self):
        TP, FN = self.matrix[0]
        FP, TN = self.matrix[1]
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self):
        TP, FN = self.matrix[0]
        FP, TN = self.matrix[1]
        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    def f1score(self):
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    def formatMatrix(matrix):
        # Convert numpy arrays to list
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
        
        # Convert torch tensors to list
        else: 
            if isinstance(matrix, torch.Tensor):
                matrix = matrix.tolist()
        
        # Validate structure
        if not isinstance(matrix, list):
            raise TypeError("Matrix must be a list, numpy.ndarray, or torch.Tensor.")
        
        if len(matrix) != 2:
            raise ValueError("Matrix must have exactly 2 rows.")
        
        for i, row in enumerate(matrix):
            if not isinstance(row, list):
                raise TypeError(f"Row {i} is not a list.")
            if len(row) != 2:
                raise ValueError(f"Row {i} must have exactly 2 elements.")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Element at position [{i}][{j}] must be an int or float.")
        
        return matrix  # return the validated and converted matrix
    
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
        
        if name[0][0]=="P":
            self.hasPD = True
        else:
            self.hasPD = False
            
        self.patientNumber = name[0]
        self.exerciseNumber = name[1]
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
    
    #def wav2vecSave(self,audiovector):
    #    self.wav2vecTensor = audiovector

    #def pyannoteSave(self,audiovector):
    #    self.pyannoteVec = audiovector

    #TODO use CM
    def saveConfusionMatrix(self,singleConfMatrix):
        self.confMatrix = singleConfMatrix
        if singleConfMatrix[0,0] == 1 or singleConfMatrix[1,1] == 1:
            self.accurate = True
        else:
            self.accurate = False
    
def getPDvecs(recordings): #takes list of recordings retruns two tensors with [hasPDvec, noPDvec]
    #TODO rename to split2PD
    hasPDvec=[]
    noPDvec=[]
    for recording in recordings:
        if recording.hasPD:
            hasPDvec.append(recording.audio2vec)
        else:
            noPDvec.append(recording.audio2vec)
    return hasPDvec, noPDvec

    
def get_all_file_paths(dir="resources/recordings"):   #filepaths = get_all_file_paths("resources/recordings")
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
    def confMatrix(self):
        confMatrix = np.array([[0,0],[0,0]])
        for recording in self.recordings:
            confMatrix+=recording.confMatrix
        return confMatrix

        
          

    def make(self,recordings):
        self.recordings = recordings
        self.patients = self.loadPatients()

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

    def loadPatients(self):
        patientList = []
        
        for recording in self.recordings:
            if  any(patient.number == recording.patientNumber for patient in patientList):
                patientList[-1].recordingsList.append(recording)
            else:
                patientList.append(Patient(recording))
                patientsCount = len(patientList)
        return patientList
    

        
    def save(self, fileObjects=None):
        if fileObjects == "files.csv":
            raise NameError("Cannot save as files.csv")           
        if fileObjects is None:
            fileObjects = self.databaseFilename
        file ="resources/databases/" + fileObjects
        #unique_filename = get_unique_filename("results.txt")
        f = open(file, "w")
        for recording in self.recordings:
            f.writelines([recording.path,SP1,recording.audio2vecStr,os.linesep])
        f.close()
        print("Database has been saved to file", fileObjects)

     #TODO:save matrixes in better format:
    def saveConfusionMatrixes(self):
        file ="resources/confusionMatrixes/" + self.name +"_CM" + ".csv"
        f = open(file, "w")
        
        for recording in self.recordings:
            if recording.accurate is None:
                ValueError("Conf. Matrix cannot be saved.")
            matrixStr = SP2.join(map(str,[item for sublist in recording.confMatrix for item in sublist]))
            f.writelines([str(recording.path),SP1,matrixStr,os.linesep])

        f.close()


    def loadConfusionMatrixes(self): #TODO rename to CMfromDatabase
        filename = self.name + "_CM" + ".csv"
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
    
    @staticmethod
    def makeRecording(recording, inference):
        filepath = recording.path
        # checking if it is a file
        if os.path.isfile(filepath):
            #sampleCount = getSampleNumber(filepath)
            #print("Samples: {n}", sampleCount)
            
            embedding = inference(filepath)
            embedding = np.median(embedding.data, axis=0)
            recording.audio2vecSave(embedding)

    def makePyannote(self):

        from pyannote.audio import Model
        from pyannote.audio import Inference

        # Load pre-trained speaker embedding model
        model = Model.from_pretrained("pyannote/embedding", use_auth_token="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", revision="main", strict=False)
        inference = Inference(model, window="sliding")
        
        
        #multithreading doesnt work  

            #from multiprocessing import Pool
            #import multiprocessing
            #cpusUsed = multiprocessing.cpu_count()-2
            
            #with Pool(processes=multiprocessing.cpu_count()-2) as pool:  # Use all available CPU cores
                #pool.starmap(self.makeRecording, [(rec, inference) for rec in self.recordings], chunksize=int(len(self.recordings)/cpusUsed))
        for recording in self.recordings:
           self.makeRecording(recording, inference)

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


    def toTensor(self, labelsON=True): 
        #returns matrix with embeddings and vector of labels (1 for has PD, -1 for CG)
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
    
    class Accuracy:
            
        
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
                print("---------------------------------------------------")

        

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

        def printspecialexcs(database,STR):
            cm = np.zeros((2, 2), dtype=int)
            for exercise in database.recordings:
                if STR in exercise.exerciseNumber:
                    cm += exercise.confMatrix
            acu = (cm[1, 1]+cm[0,0])
            racy= (cm[1, 1]+cm[0,0]+cm[1, 0]+cm[0, 1])
            acuracy = acu/racy
            print("---------------------------------------------------")
            print(STR," ex. confusion Matrix:")
            print(f"TP: {cm[1, 1]}", end="  ")
            print(f"FN: {cm[1, 0]}")
            print(f"FP: {cm[0, 1]}", end="  ")
            print(f"TN: {cm[0, 0]}")
            print(f"ex. accuracy is {acuracy:.2f} ({acu}/{racy})")

        def printMetrics(database):
            confusionMatrix = database.confMatrix()
            print("---------------------------------------------------")
            print("Confusion Matrix:")
            print(f"TP: {confusionMatrix[1, 1]}", end="  ")
            print(f"FN: {confusionMatrix[1, 0]}")
            print(f"FP: {confusionMatrix[0, 1]}", end="  ")
            print(f"TN: {confusionMatrix[0, 0]}")

            acuracy =  (confusionMatrix[1,1] + confusionMatrix[0, 0])/(confusionMatrix[1,1] + confusionMatrix[0, 1]+confusionMatrix[1,0] + confusionMatrix[0,0])
            precision = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[0, 1])
            recall = confusionMatrix[1,1] / (confusionMatrix[1,1] + confusionMatrix[1, 0])
            f1_scored = 2 * (precision * recall) / (precision + recall)

            print("---------------------------------------------------")
            print("TOTAL")
            print(f"Acurracy: {100*acuracy:.2f}")
            print(f"Precision: {100*precision:.2f}")
            print(f"Recall: {100*recall:.2f}")
            print(f"F1 Score: {100*f1_scored:.2f}")
            print("---------------------------------------------------")


            
 
def getSampleNumber(filepath):
        with wave.open(filepath, "r") as wf:
            num_samples = wf.getnframes()
        return num_samples

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
                print(f"Padded Duration: {len(y) / sr} seconds!!!!!!!!!!!!!!!!!!!!")

                # Save the padded file temporarily
                temp_filepath = filepath.replace(".wav", "_padded.wav")
                sf.write(temp_filepath, y, sr)
                filepath = temp_filepath  # Update filepath to padded file



#from multiprocessing import freeze_support
#if __name__ == '__main__':
#    freeze_support()  # Needed for Windows/macOS
#    pyannoteDB = Database()
#    pyannoteDB.makePyannote()
#    pyannoteDB.save(fileObjects="pyannote.csv")
#    pyannoteDB = Database()
#    pyannoteDB.load("pyannote.csv")
#    pyannoteDB.loadConfusionMatrixes()
#    print(pyannoteDB.recordings[0].accuracy)
