# import librosa
# import numpy as np
# from scipy.io import wavfile

# SAMPLING_RATE = 16000
# INPUT_LENGTH = 9.01

# def audio_melspec(fpath, sampling_rate=SAMPLING_RATE, n_mels=120, frame_size=320, hop_length=160):
#     # Load and normalize audio
#     input_fs, aud = wavfile.read(fpath)
#     aud = aud.astype(np.float32) / np.iinfo(aud.dtype).max

#     # Resample if needed
#     if input_fs != sampling_rate:
#         aud = librosa.resample(y=aud, orig_sr=input_fs, target_sr=sampling_rate)

#     # Ensure minimum length
#     len_samples = int(INPUT_LENGTH * sampling_rate)
#     while len(aud) < len_samples:
#         aud = np.append(aud, aud)

#     # Compute Mel spectrogram
#     mel_spec = librosa.feature.melspectrogram(y=aud[:len_samples], sr=sampling_rate, 
#                                               n_fft=frame_size+1, hop_length=hop_length, 
#                                               n_mels=n_mels)
#     mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40  # Normalize
#     mel_spec = mel_spec.T  # Transpose for model input

#     return mel_spec.tolist()  # Convert to list (JSON serializable)


import librosa
import numpy as np
from scipy.io import wavfile

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

def get_melspec(fpath, sampling_rate=SAMPLING_RATE):
    input_fs, aud = wavfile.read(fpath)
    aud = aud.astype(np.float32) / np.iinfo(aud.dtype).max

    if input_fs != sampling_rate:
        audio = librosa.resample(y=aud, orig_sr=input_fs, target_sr=sampling_rate)
    else:
        audio = aud

    len_samples = int(INPUT_LENGTH * sampling_rate)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)

    num_hops = int(np.floor(len(audio) / sampling_rate) - INPUT_LENGTH) + 1
    hop_len_samples = sampling_rate
    all_melspecs = []

    for idx in range(num_hops):
        start = int(idx * hop_len_samples)
        end = int((idx + INPUT_LENGTH) * hop_len_samples)
        audio_seg = audio[start:end]
        
        if len(audio_seg) < len_samples:
            continue

        input_features = np.array(audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
        all_melspecs.append(input_features.tolist()[0])

    return all_melspecs  # List of segments' mel specs

def audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
    if to_db:
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
    return mel_spec.T