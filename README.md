# Speech Denoising Using Deep Learning
This project implements a deep learning-based speech denoising system using a fully-connected neural network. The goal is to remove noise from a speech signal contaminated with chip-eating noise.

Dataset
The dataset consists of the following audio files:

train_clean_male.wav: Clean speech signal used for training the model.
train_dirty_male.wav: Noisy speech signal with chip-eating noise, used for training the model.
test_x_01.wav: Noisy speech signal used for validation.
test_x_02.wav: Noisy speech signal containing Professor K's voice contaminated with chip-eating noise, used for testing.
test_s_01.wav: Clean speech signal corresponding to test_x_01.wav, used for computing the Signal-to-Noise Ratio (SNR).

Implementation:
The code loads the audio files, converts them to spectrograms using the Short-Time Fourier Transform (STFT), and extracts the magnitude spectra. A fully-connected neural network with two hidden layers is trained to predict the clean magnitude spectra from the noisy magnitude spectra.

The network architecture consists of the following layers:
Input layer with 513 units
First hidden layer with 256 units and ReLU activation
Second hidden layer with 128 units and ReLU activation
Output layer with 513 units and ReLU activation

The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function.
After training, the model is used to predict the clean magnitude spectra for the validation and test signals. The clean time-domain signals are reconstructed using the inverse STFT with the predicted clean magnitude spectra and the phase information from the noisy signals.
The Signal-to-Noise Ratio (SNR) is computed for the denoised validation signal by comparing it with the ground truth clean signal (test_s_01.wav). The denoised audio for test_x_02.wav is saved as denoised_output.wav.

Results
The code includes visualizations of the noisy speech spectrogram, recovered speech spectrogram, and clean speech spectrogram. Additionally, the audio files for the noisy speech, recovered speech, and clean speech are played.
The achieved SNR for the denoised validation signal is reported in the output.
