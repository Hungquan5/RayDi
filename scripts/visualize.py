import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file
audio_file = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/HOMAI/RayDi_TheSceretary/Voice_Assitant/data_test/0_resample/recording_20241127_145001_16k.wav'
y, sr = librosa.load(audio_file, sr=16000)
# y, sr = librosa.load(file_path, sr=16000)
y = librosa.util.normalize(y)
mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13, n_fft=1024, hop_length=512, window='hann')
mfcc = mfcc.T  # Transpose so that time is the first dimension
mfcc = (mfcc - np.mean(mfcc, axis=0)) / (np.std(mfcc, axis=0) + 1e-8)

# Step 1: Frame and Window the Signal
# Handled by librosa in MFCC calculation

# Step 2: Extract MFCC Features
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512, n_fft=2048)
# print(mfccs.shape)
# input()
# Step 3: Add Deltas and Delta-Deltas (Temporal Derivatives)
delta_mfccs = librosa.feature.delta(mfcc)
delta2_mfccs = librosa.feature.delta(mfcc, order=2)

# Step 4: Spectral Features
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Step 5: Energy and Zero-Crossing Rate (ZCR)
rms = librosa.feature.rms(y=y)
zcr = librosa.feature.zero_crossing_rate(y)

# Step 6: Combine Features into a Single Feature Vector
# combined_features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs, spectral_contrast, chroma, rms, zcr), axis=0)
# print (combined_features.shape)
# input()
# Step 7: Save Example Features as Images

# Save MFCCs as an image
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfcc, x_axis='time', sr=sr)
plt.colorbar(format='%+2.0f dB')
plt.title('MFCCs')
plt.savefig('mfccs_image3.png')  # Save MFCC plot as an image
plt.close()

plt.figure(figsize=(10, 6))
librosa.display.specshow(spectral_contrast, x_axis='time', sr=sr)
plt.colorbar()
plt.title('Spectral Contrast')
plt.savefig('spectral_contrast_image3.png') 
# Visualize the raw waveform
plt.figure(figsize=(10, 6))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform of the Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('waveform_image3.png')  # Save the waveform plot as an image
plt.close()  # Close the figure
