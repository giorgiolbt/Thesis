import audio_loader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



#Assigns the analyzed_audios to the variable
prescaled_data = (audio_loader.process_audio_files(audio_loader.audios)).astype(np.float64)
labels = audio_loader.labels
#These are the possible categories of emotions
emotions = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutrality', 'Sadness', 'Surprise']

#The corpus is defined as a dictionary where the keys will be the categories of emotions
#while the values are list of samples analyzed (for that emotion). So it will be a list of arrays because we have multiple features for each sample
corpus = {}

for em in emotions:
    corpus[em] = []


# Preprocessing of the data --> wrong: it must be done once I've done the split in training and test. Actually I don't know: because sometimes it says that it's better to do like this
standardized_data = preprocessing.scale(prescaled_data)

#standardized_data.std(axis=0)

#this loop populates the corpus
for index in range(len(labels)):
    if labels[index] == 'Anger':
        corpus['Anger'].append(standardized_data[index])
    elif labels[index] == 'Disgust':
        corpus['Disgust'].append(standardized_data[index])
    elif labels[index] == 'Fear':
        corpus['Fear'].append(standardized_data[index])
    elif labels[index] == 'Joy':
        corpus['Joy'].append(standardized_data[index])
    elif labels[index] == 'Neutrality':
        corpus['Neutrality'].append(standardized_data[index])
    elif labels[index] == 'Sadness':
        corpus['Sadness'].append(standardized_data[index])
    elif labels[index] == 'Surprise':
        corpus['Surprise'].append(standardized_data[index])

#print(len(corpus['Surprise']))
#print(corpus['Anger'])

#Here the training and test sets are built. They're dictionaries where the key is the emotion and the value is a list of data (list of the arrays with the features of each sample)
train_set = {}
test_set = {}

for em in emotions:
    train_set[em], test_set[em] = train_test_split(corpus[em], test_size=0.15, random_state=42)

print(train_set['Anger'])



