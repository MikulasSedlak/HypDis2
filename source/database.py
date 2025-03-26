import os
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
        self.audio2vec = audiovector
        audiovector = audiovector.tolist()
        self.audio2vecStr = SP2.join(map(str,audiovector))
    
    #def wav2vecSave(self,audiovector):
    #    self.wav2vecTensor = audiovector

    #def pyannoteSave(self,audiovector):
    #    self.pyannoteVec = audiovector

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



def loadConfusionMatrix(databaseFilename): #TODO rename to CMfromDatabase
    print("loading ", databaseFilename, "...")
    databasePath ="resources/Confusion_matrixes/" + databaseFilename
    recordings = []
    with open(databasePath, "r") as openfileobject:
        for line in openfileobject:

            #splits into path
            linestr = line.strip()
            linestr= linestr.split(SP1)

            #creates recording
            recordings.append(Recording(linestr[0]))
            
            #create and save Conf matrix
            linestr = linestr[1].split(SP2)
            linestr.pop(0)
            recordings[-1].saveConfusionMatrix([[int(linestr[0])],[linestr[1]]],[[linestr[2]],[linestr[3]]])

    print(databasePath," succesfully loaded.")

def saveConfusionMatrix(filename, fileObjects):
    file ="resources/confusionMatrixes/" + filename
    f = open(file, "w")
    for recording in fileObjects:
      matrixStr = SP2.join(map(str,recording.confMatrix))
      f.writelines([str(recording.path),SP1,matrixStr,os.linesep])
    f.close()




class Database:
    count = 0
    def __init__(self,databaseFilename="files.csv"):
        self.recordings = self.load(databaseFilename)
        self.databaseFilename = databaseFilename
        self.count = 0
    @staticmethod
    def countFunc():
        Database.count +=1
        print("count: ", Database.count)
    

    def load(self,databaseFile):
        print("loading ", databaseFile, "...")
        databasePath ="resources/databases/" + databaseFile
        recordings = []
        with open(databasePath, "r") as openfileobject:
            for line in openfileobject:
                recordings.append(Recording(line.strip()))
        print(databasePath," loaded.")
        return recordings
    
    def save(self, fileObjects=None):
        if fileObjects == "files.csv":
            raise NameError("Cannot save as files.csv")           
        if fileObjects is None:
            fileObjects = self.databaseFilename
        file ="resources/databases/" + fileObjects
        f = open(file, "w")
        for recording in self.recordings:
            f.writelines([recording.path,SP1,recording.audio2vecStr,os.linesep])
        f.close()
        print("Database has been saved to file", fileObjects)


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
    def processRecording(recording, inference):
        filepath = recording.path
        # checking if it is a file
        if os.path.isfile(filepath):
            #sampleCount = getSampleNumber(filepath)
            #print("Samples: {n}", sampleCount)
            
            embedding = inference(filepath)
            embedding =  np.median(embedding.data, axis=0)

            recording.audio2vecSave(embedding)
            #Database.countFunc()

    def makePyannote(self):

        from pyannote.audio import Model
        from pyannote.audio import Inference

        # Load pre-trained speaker embedding model
        model = Model.from_pretrained("pyannote/embedding", use_auth_token="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")
        inference = Inference(model, window="sliding")
        
        import os
        import numpy as np
        from multiprocessing import Pool
        import multiprocessing
        
        
        #for recording in self.recordings:
        with Pool(processes=multiprocessing.cpu_count()-2) as pool:  # Use all available CPU cores
            
            #results = pool.starmap(self.processRecording, [(rec, inference) for rec in self.recordings])
                results = pool.starmap(self.processRecording, [(rec, inference) for rec in self.recordings], chunksize=16)

 
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



from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()  # Needed for Windows/macOS
    pyannoteDB = Database()
    pyannoteDB.makePyannote()
    pyannoteDB.save("pyannote.csv")
