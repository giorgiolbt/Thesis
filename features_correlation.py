import audio_loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#here result_dataframe is a list of the samples each one as a pandas df with num of features columns
result_dataframe = audio_loader.process_audio_files(audio_loader.audios, False, True)


#calculates the correlation between each couple of features and plots a heatmap. The 'dataframe' must be a pandas dataframe
def correlation_heatmap(dataframe):
    correlations = dataframe.corr()

    fig, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()



#this function calculates the variables that are correlated and returns a copy of the dataframe that excludes those correlated
# variables. The threshold to decide the correlation is passed as a parameter
def correlation(dataset, threshold):
    deep_copy = dataset.copy()
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = deep_copy.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            #TODO: assicurati che magari abbia più senso prendere il valore assoluto
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in deep_copy.columns:
                    del deep_copy[colname] # deleting the column from the dataset
    #print(len(col_corr))
    #print(sorted(col_corr))
    return deep_copy

def append_target(df, target, column):
    data_frame_with_target = df.copy()
    data_frame_with_target[target] = audio_loader.labels
    return data_frame_with_target


#to obtain the actual n_samples x n_features data structure
data_frame = pd.concat(result_dataframe)

#correlation_heatmap(data_frame)

#adds the column with the target variable to the dataframe
data_frame_complete = append_target(data_frame, 'Emotion', audio_loader.labels)
#convert the target to a numerical value TODO: I don't know if it actually works (the correlation values obtained are quite low)
data_frame_complete['Emotion'] = data_frame_complete['Emotion'].astype('category')
data_frame_complete["Emotion_cat"] = data_frame_complete["Emotion"].cat.codes
#computes the correlation of each feature with the target variable
print(data_frame_complete.corrwith(data_frame_complete["Emotion_cat"]))

