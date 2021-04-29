# load audios'paths and labels in two lists

import glob
audios = []
labels = []
#specify the appropriate path to find the files
for filepath in glob.iglob('/Users/giorgiolabate/Desktop/Polimi/COMPUTER SCIENCE AND ENGINEERING/III ANNO/Thesis/emovo_it/EMOVO_italiano/*/*/*.wav', recursive=True):
    audios.append(filepath)
    if "Anger" in filepath:
        labels.append("Anger")
    elif "Disgust" in filepath:
        labels.append("Disgust")
    elif "Fear" in filepath:
        labels.append("Fear")
    elif "Joy" in filepath:
        labels.append("Joy")
    elif "Neutrality" in filepath:
        labels.append("Neutrality")
    elif "Sadness" in filepath:
        labels.append("Sadness")
    elif "Surprise" in filepath:
        labels.append("Surprise")


import opensmile
import audiofile
import pandas as pd
import numpy as np


# analyze the audios through Opensmile and return the processed_audios as a list of numpy arrays
def process_audio_files(audios, extended):  # audios is the list of the audio files' paths
    processed_audios = []
    if (extended):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    else:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    for audio in audios:
        print(audio)
        signal, sampling_rate = audiofile.read(audio, always_2d=True)
        result = smile.process_signal(
            signal,
            sampling_rate
        )
        processed_audios.append(result.values.astype(np.float32))
        # processed_audios.append(result)# usato per fare l'analisi di correlazione
    return processed_audios

result = process_audio_files(audios, False)
prescaled_data = result

maxim = len(max(prescaled_data, key=len))
for i in range(len(prescaled_data)):
    offset = prescaled_data[i].shape[0]
    padding = maxim - offset
    prescaled_data[i] = np.concatenate((prescaled_data[i], np.zeros((padding,18))), axis = 0)

lengths = []
for el in prescaled_data:
    lengths.append(len(el))

#concatenate for the feature normalization/standardization
prescaled_data_concat = np.concatenate(prescaled_data)


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, RobustScaler

#here I normalize (between 0 and 1, NO MORE negative values)
scaler = MinMaxScaler()
scaler.fit(prescaled_data_concat)

scaled_data = scaler.transform(prescaled_data_concat)

#used to split back the samples after feature normalizatin
lengths_cum = np.cumsum(lengths)
scaled_data_final = np.split(scaled_data, lengths_cum)
scaled_data_final = scaled_data_final[:588]

#corpus is a dictionary where the emotion is the key and the values will be the associated samples.
emotions = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutrality', 'Sadness','Surprise' ]
corpus = {}
for em in emotions:
    corpus[em] = []

#populates the corpus

for index in range(len(labels)):

    if labels[index] == 'Anger':
        corpus['Anger'].append(scaled_data_final[index])
    elif labels[index] == 'Disgust':
        corpus['Disgust'].append(scaled_data_final[index])
    elif labels[index] == 'Fear':
        corpus['Fear'].append(scaled_data_final[index])
    elif labels[index] == 'Joy':
        corpus['Joy'].append(scaled_data_final[index])
    elif labels[index] == 'Neutrality':
        corpus['Neutrality'].append(scaled_data_final[index])
    elif labels[index] == 'Sadness':
        corpus['Sadness'].append(scaled_data_final[index])
    elif labels[index] == 'Surprise':
        corpus['Surprise'].append(scaled_data_final[index])

train_set = {}
test_set = {}
from sklearn.model_selection import train_test_split

for em in emotions:
    train_set[em], test_set[em] = train_test_split(corpus[em], test_size=0.15, random_state=1)






from kesmarag.hmm import HiddenMarkovModel, new_left_to_right_hmm, store_hmm, restore_hmm, toy_example

#in questo modo ottengo il formato per la nuova libreria da provare
#newarray = np.dstack(train_set['Anger'])
#newarray = np.rollaxis(newarray,-1)

#print(newarray.shape)
#model = new_left_to_right_hmm(states=3, mixtures=2, data=np.expand_dims(train_set['Anger'][0], axis=0))
#model.fit(newarray, verbose=True)

#newarray_test = np.dstack(test_set['Anger'])
#newarray_test = np.rollaxis(newarray_test,-1)

#print(model.log_posterior(newarray_test))

from hmmlearn import hmm

# Models saved as a dictionary where the keys are the emotions and the value the corresponding classifier
hmms = {}


def train_hmms():
    for em in emotions:
        print("training for emotion:", em)
        hmms[em] = new_left_to_right_hmm(states=2, mixtures=2, data=np.expand_dims(train_set[em][0], axis=0))
        newarray = np.dstack(train_set[em])
        newarray = np.rollaxis(newarray, -1)
        hmms[em].fit(newarray, verbose = True)


train_hmms()

def test_hmms():
    # Dictionary that will contain the accuracy of each classifier
    accuracies = {}
    total = 0
    correct = 0

    #analyzes the emotions one at a time
    for em in emotions:
        print("ANALYZED EMOTION: " + em + "\n")
        total_specific_em = 0
        correct_specific_em = 0
        correct_label = em
        #analyzes each sample in the test set of the emotion analyzed
        for test_sample in test_set[em]:
            best_res = -float('inf')
            predicted_emotion = None
            #calculates the likelihood (of a sample being produced by a HMM) produced by each one of the 7 classifiers
            for hmm in emotions:
                temp = hmms[hmm]
                #the exponential is calculated since the 'score' function calculates the log likelihood
                score = temp.log_posterior(np.expand_dims(test_sample, axis=0))
               # print(score)
                if (score > best_res):
                    best_res = score
                    predicted_emotion = hmm
            #print("The CORRECT label for this sample was: ", correct_label)
            #print("The PREDICTED label for this sample is: ", predicted_emotion)
            if (predicted_emotion == correct_label):
                correct += 1
                correct_specific_em += 1
            total += 1
            total_specific_em += 1
        print("The accuracy of the classifier for the emotion " + em + " is:",
              (correct_specific_em / total_specific_em) * 100)
        accuracies[em] = (correct_specific_em/total_specific_em)*100
    print("\nThe micro accuracy of the classifier is: ", (correct / total) * 100)
    #print("The macro accuracy of the classifier is: ", array([accuracies[em] for em in accuracies]).mean())

test_hmms()