import torch 
import matplotlib.pyplot as plt
import pandas as pd
import closerVector as cV
import database as DB
import numpy as np


def boxplot(databases,use_median=True):
    xticks = []
    #loads databases
    for i, database in enumerate(databases):


        #median or average
        if use_median:
            P, K = cV.medianVec(database.recordings)
            method = "median"
        else:
            P, K = cV.avgVec(database.recordings)
            method = "average"
        
        #normalise
        scale = 1.0 / torch.norm(K - P)

        #normalise recordings
        for recording in database.recordings:
            recording.audio2vec = recording.audio2vec * scale

        #normalise vectors
        K = K * scale
        P = P * scale

        pd_diffs = [
            cV.absDist(K - r.audio2vec) - cV.absDist(P - r.audio2vec)
            for r in database.recordings if r.hasPD
        ]
        nonpd_diffs = [
            cV.absDist(K - r.audio2vec) - cV.absDist(P - r.audio2vec)
            for r in database.recordings if not r.hasPD
        ]

        distanceKFromP_P = pd_diffs
        distanceKFromP_K = nonpd_diffs

            
        plt.boxplot(distanceKFromP_P, positions=[2 * i + 1], patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='blue', color='black'), medianprops=dict(color='black'))
        plt.boxplot(distanceKFromP_K, positions=[2 * i + 2], patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor='red', color='black'), medianprops=dict(color='black'))
        xticks.append(database.name)
        xticks.append("")
    plt.xticks(list(range(1, len(xticks)+1)), xticks)
    
    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(facecolor='blue', edgecolor='black', label='PD'),
        Patch(facecolor='red', edgecolor='black', label='no PD')
    ])
    plt.title('Vzdálenost od průměrných vektorů')
    plt.grid(axis='y')
    plt.yticks([-1, 0,1], ['bez HD',"", 'HD'])
    plt.show()
    #plt.savefig("boxplots/" + "boxplot_avg.png")

def ratioOfLen():
     for path in DB.loadpaths:
        database = DB.Database()
        database.load(path)
        recordings = database.recordings
        
        sum1 = []
        for recording in recordings:
            sum1.append(np.linalg.norm(recording.audio2vec))
        avg = np.median(sum1)

        P, K = cV.medianVec(recordings)
        PKDist = np.linalg.norm(K-P)
        print(f"ratio dist/avgvec = {PKDist/avg}")

if __name__ == "__main__":
    databases = DB.loadAllDB()
    boxplot(databases)