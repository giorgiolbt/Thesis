import opensmile
import audiofile #This package can 'read' audio files

audio = '/Users/giorgiolabate/Downloads/opensmile-3.0-osx-x64/example-audio/opensmile.wav'


def extract_functionals(audio):
    # the 'read' function returns a two-dimensional array in the form [channels, samples]. If the sound file has only one channel, a one-dimensional array is returned
    # it also returns the sample rate of the audio file
    # the 'always_2d if True it always returns a two-dimensional signal even for mono sound files
    signal, sampling_rate = audiofile.read(audio, always_2d=True)

    # create a feature extractor for functionals features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # extract features for the signal. It returns the DataFrame
    result = smile.process_signal(
        signal,
        sampling_rate
    )

    return result

def extract_LLD(audio):
    # the 'read' function returns a two-dimensional array in the form [channels, samples]. If the sound file has only one channel, a one-dimensional array is returned
    # it also returns the sample rate of the audio file
    # the 'always_2d if True it always returns a two-dimensional signal even for mono sound files
    signal, sampling_rate = audiofile.read(audio, always_2d=True)

    # create a feature extractor for LLD features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    # extract features for the signal. It returns the DataFrame
    result = smile.process_signal(
        signal,
        sampling_rate
    )

    return result

def extract_LLD_deltas(audio):
    # the 'read' function returns a two-dimensional array in the form [channels, samples]. If the sound file has only one channel, a one-dimensional array is returned
    # it also returns the sample rate of the audio file
    # the 'always_2d if True it always returns a two-dimensional signal even for mono sound files
    signal, sampling_rate = audiofile.read(audio, always_2d=True)

    # create a feature extractor for LLD deltas features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
    )

    # extract features for the signal. It returns the DataFrame
    result = smile.process_signal(
        signal,
        sampling_rate
    )

    return result