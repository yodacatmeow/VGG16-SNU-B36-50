"""
Descriptions
    # Convert a given audio file to a log scaled Mel-spectrogram size of 224 x 224
    # Time length of the log scaled mel-spectrogram is "time_len" [sec]
    # Since input of VGG 16 is 224 x 224 x 3, the code stack 3 patches
References
    # Grey img array to RGB img array
        http://www.socouldanyone.com/2013/03/converting-grayscale-to-rgb-with-numpy.html
"""

# Public python modules
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

# Audio path
def get_audio_path(audio_dir, track_id):
    audio_format = '.m4a'
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + audio_format)

# Patch time-limit (Automatic)
def patch_lim(s_n, fs, time_len, n_mels, n_width, n_fft, hop_length):
    E = librosa.feature.melspectrogram(y=s_n, sr=fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=int(fs / 2), power=1)
    E_col_sum = np.sum(E, axis=0)  # Column sum of "E"
    E_area_sum = np.zeros(len(E_col_sum) - n_width + 1)  # initialize array
    # Calculate sum of E[i,j] in an area using "E_col_sum"
    for i in range(len(E_area_sum)):
        E_area_sum[i] = np.sum(E_col_sum[i:i + n_width])
    # Extract a patch from "E" size of "n_mel" x "n_width" with the maximum area sum
    index = [np.argmax(E_area_sum), np.argmax(E_area_sum) + n_width]
    # If index start is negative
    if index[0] < 0:
        index = [0, n_width]
    else:
        pass
    # Patch time-limit index; Unit=sample
    index = [hop_length * index[0], int(hop_length * index[0] + fs * time_len)]
    return index

# Patch time limit (Refer 'event_start' in the metadata)
def patch_lim_meta(s_n, fs, event_start, time_len, offset = 0.1):
    index = [int(event_start - offset*fs), int(event_start + fs * (time_len - offset))]
    if index[1] > len(s_n):
        index = [int(len(s_n) - fs*time_len), len(s_n)]
        print("index[1] exceeds len(s_n)")
    else:
        pass
    return index

# Mel-spectrogram; returns [Mel-spec, Mel-spec, Mel-spec]; Generate a log scaled Mel-spectrogram using librosa.melspectrogram
def melspec(s_n, fs, n_fft, hop_length, n_mels, fmax):
    n_width = n_mels
    mel_spec = librosa.feature.melspectrogram(y=s_n, sr=fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax, power=2)
    #_ To dB
    mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
    print(mel_spec.shape)
    #_ Stack three patches to make an input dimension of n_width x n_mels x 3
    patch_rgb = np.empty((n_width, n_mels, 3), dtype = np.float32)
    patch_rgb[:, :, 0] = mel_spec
    patch_rgb[:, :, 1] = mel_spec
    patch_rgb[:, :, 2] = mel_spec
    return patch_rgb

# Mel-spectrogram; returns [Mel-spec, Mel-spec, Mel-spec]
# Generate Mel-filterbank and multiply the filterbank with a spectrogram
def melspec2(s_n, fs, n_fft, win_length, hop_length, n_mels, fmax, power):
    n_width = n_mels
    #- Spectrogram; dtype = complex64
    S = librosa.core.stft(y=s_n, win_length=win_length, n_fft=n_fft, hop_length=hop_length)
    #- To power
    S = np.abs(S)**power
    #- Mel-filterbank
    fb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=fmax, norm=None)
    #- To Mel-spectrogram
    S = np.matmul(fb, S)
    #- To dB
    S = librosa.logamplitude(S, ref_power=np.max)
    #- Prepare return
    P = np.empty((n_width, n_mels, 3), dtype=np.float32)
    P[:, :, 0] = S
    P[:, :, 1] = S
    P[:, :, 2] = S
    return P

# Spectrogram; reshaped bands
def spec_reshape(s_n, win_length, n_mels, n_fft, hop_length):
    patch_stft = librosa.core.stft(y=s_n, win_length=win_length, n_fft=n_fft, hop_length=hop_length)
    patch_stft = np.abs(patch_stft) ** 2
    patch_stft = librosa.logamplitude(patch_stft, ref_power=np.max)  # To dB
    patch_stft = patch_stft[0:4 * n_mels, :]
    patch_stft = patch_stft[0::4] + patch_stft[1::4] + patch_stft[2::4] + patch_stft[3::4]
    return patch_stft


# Feature generation
def feature(tid, event_start=0):
    try:
        # Size of feature
        n_mels = 224; n_width = n_mels

        # 3.0 sec 224x224 patch @fs=44,100Hz; n_fft = window_size=2048 and hop_size=592 gives 224 x 224 log scaled mel-spectrogram
        time_len = 3; win_length = 2048; n_fft = win_length; hop_length = 592

        # 1.5 sec 224x224 patch @fs=44,100Hz; n_fft = window_size=2048 and hop_size=296 gives 224 x 224 log scaled mel-spectrogram
        #time_len = 1.5; win_length = 2048; n_fft = win_length; hop_length = 296

        # 1.5 sec 224x224 patch @fs=44,100Hz; n_fft = 2048, window size=1024 and hop_size=296 gives 224 x 224 mel-spectrogram
        #time_len = 1.5; win_length = 1024; n_fft = 2048; hop_length = 296;

        # File path
        filepath = get_audio_path('audio', tid)

        # Read audio files
        s_n, fs = librosa.load(filepath, sr=None, mono=True)  # s_n = signal, fs = sampling freq.
        # If audio input is shorter than "time_len", then pads zeros
        if len(s_n) < time_len * fs:
            s_n = np.pad(s_n, (0, time_len * fs - len(s_n)), 'constant')
            print("sample is shorter than time_len [tid]:", tid)

        # Patch time limit (when mode = auto)
        #patchlim = patch_lim(s_n, fs, time_len, n_mels, n_width, n_fft, hop_length)

        # Patch time limit (when mode = refer metadata)
        patchlim = patch_lim_meta(s_n, fs, event_start, time_len)

        # Cut 's_n'
        s_n = s_n[patchlim[0]:patchlim[1]]

        # Patch
        patch = melspec2(s_n=s_n, fs=fs, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, fmax=int(fs/2), power=2)
        #patch = spec_reshape(s_n, win_length, n_mels, n_fft, hop_length)


    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        return False, 0

    return True, patch

if __name__ == "__main__":
    # Test0; This will returns: "dataset/train/054/054151.m4a"
    #print(get_audio_path('audio', 54151))

    # Test 1; e.g. feature(10049) draws "dataset/train/010/010049.m4a"
    result, feature = feature(38001, 41440); print(feature.shape)

    # Test 2; Draw a spectrum img; You need to return "patch_xx" instead of "rgb_patch"
    plt.figure(figsize=(8, 6)), librosa.display.specshow(feature[:,:,0], x_axis='time'), plt.colorbar(), plt.clim(np.min(feature), np.max(feature)), plt.show()

    # Test 3; BGR mode
    #plt.imshow(feature[:,:,0]); plt.show()
