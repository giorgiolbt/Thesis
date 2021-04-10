import audio_loader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



#Assigns the analyzed_audios to the variable
#TODO: control if the rounding of the floats is ok
prescaled_data = (np.genfromtxt('/Users/giorgiolabate/PycharmProjects/Thesis/analyzed_audios.csv', delimiter = ',')).astype(np.float32)
labels = np.genfromtxt('/Users/giorgiolabate/PycharmProjects/Thesis/labels.csv', delimiter = ',', dtype= str)
#These are the possible categories of emotions
emotions = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutrality', 'Sadness', 'Surprise']

#This funtion load the corpus of audio data. The corpus is defined as a dictionary where the keys will be the categories of emotions
#while the values are list of samples analyzed (for that emotion). So it will be a list of arrays because we have multiple features for each sample.
#If standardize is set to 'True' data will be scaled otherwise they won't
def load_corpus(standardize):

    corpus = {}
    for em in emotions:
        corpus[em] = []

    if(standardize):
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
    else:
        # this loop populates the corpus
        for index in range(len(labels)):
            if labels[index] == 'Anger':
                corpus['Anger'].append(prescaled_data[index])
            elif labels[index] == 'Disgust':
                corpus['Disgust'].append(prescaled_data[index])
            elif labels[index] == 'Fear':
                corpus['Fear'].append(prescaled_data[index])
            elif labels[index] == 'Joy':
                corpus['Joy'].append(prescaled_data[index])
            elif labels[index] == 'Neutrality':
                corpus['Neutrality'].append(prescaled_data[index])
            elif labels[index] == 'Sadness':
                corpus['Sadness'].append(prescaled_data[index])
            elif labels[index] == 'Surprise':
                corpus['Surprise'].append(prescaled_data[index])

    return corpus


#print(len(corpus['Surprise']))
#print(corpus['Anger'])

#This function takes the corpus and the list of emotions, builds a training and test set for each emotion and returns them.
#The data structures are dictionaries where the key is the emotion and the value is a list of data (list of the arrays with the features of each sample)
def build_train_test(corpus, emotions):
    train_set = {}
    test_set = {}
    #Here the training and test sets are built.
    for em in emotions:
        train_set[em], test_set[em] = train_test_split(corpus[em], test_size=0.15, random_state=42)
    return train_set, test_set
#print(train_set['Anger'])



