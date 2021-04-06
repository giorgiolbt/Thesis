from pydub import AudioSegment

#  sound is a pydub.AudioSegment, silence_threshold in dB, chunk_size in ms
def detect_leading_silence(sound, silence_threshold, chunk_size=10):
    '''

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    #It returns the number of milliseconds of silence at the beginning of 'sound'
    return trim_ms

#This function actually removes the leading and ending silence of an audio (passed as parameter to the function),
#silence_threshold measures how much to reduce the sound loudness to find the silence threshold in dBFS
def trim_audio(sound, silence_threshold):
    start_trim = detect_leading_silence(sound, sound.dBFS*silence_threshold)
    end_trim = detect_leading_silence(sound.reverse(), sound.dBFS*silence_threshold)
    duration = len(sound)
    trimmed_sound = sound[start_trim:duration-end_trim]

    return trimmed_sound
