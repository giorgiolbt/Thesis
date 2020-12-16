import glob
import opensmile
import audiofile
import numpy as np

#I add in audios all the audiopaths and in labels the corresponding label to each audio file
#Now this is temporary to try algorithms on emovo
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


#Given the list of audio files' paths, it returns the matrix containing all the audios and the related features
#of shape n_audios x n_features
def process_audio_files(audios):  # audios is the list of the audio files' paths
    processed_audios = []
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    for audio in audios:
        signal, sampling_rate = audiofile.read(audio, always_2d=True)
        result = smile.process_signal(
            signal,
            sampling_rate
        )
        processed_audios.append(result.values[0])

    return np.array(processed_audios)

#Just a check
print(process_audio_files(audios))
