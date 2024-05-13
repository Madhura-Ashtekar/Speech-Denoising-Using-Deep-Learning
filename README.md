# Problem 1
# Speech Denoising Using Deep Learning

This project implements a deep learning-based speech denoising system using a fully-connected neural network. The goal is to remove noise from a speech signal contaminated with chip-eating noise.

Dataset:
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

Results:
The code includes visualizations of the noisy speech spectrogram, recovered speech spectrogram, and clean speech spectrogram. Additionally, the audio files for the noisy speech, recovered speech, and clean speech are played.
The achieved SNR for the denoised validation signal is reported in the output.

# Problem 2
# Speech Denoising Using 1D CNN

Overview:
This project implements a speech denoising system using a 1D Convolutional Neural Network (CNN). The objective is to remove noise from speech signals contaminated with chip-eating noise.

Model Architecture:
The neural network model is defined with Conv1D layers using the Keras functional API. The architecture includes:

Input layer: Input shape is (513, 1).
Convolutional layers: Two Conv1D layers with ReLU activation functions, followed by flattening.
Output layer: Dense layer with 513 units and ReLU activation.
Training
The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function. The training process includes 300 epochs with a batch size of 32.

Results:
SNR Improvement: The achieved Signal-to-Noise Ratio (SNR) for the denoised validation signal is reported to be 11.27 dB, indicating a significant improvement in audio quality compared to the noisy input. Further optimization could potentially raise the SNR to 15 dB.
Visualizations and Audio Playback
Spectrogram Visualization: Spectrograms of the noisy speech, recovered speech, and clean speech are plotted for comparison.
Audio Playback: Audio files for the noisy speech, recovered speech, and clean speech are played.

# Problem 3
# Data Augmentation for CIFAR-10 Classification

Overview:
This code snippet demonstrates the application of data augmentation techniques to enhance the performance of Convolutional Neural Network (CNN) models on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to classify these images into their respective categories.

Code Structure:

Data Loading and Preprocessing: 
The CIFAR-10 dataset is loaded and preprocessed. Data normalization is performed to rescale pixel values to the range [-1, 1].
Baseline CNN Model: 
A baseline CNN model is defined, consisting of convolutional layers, max-pooling layers, flatten layer, fully connected layers with ReLU activation, and a softmax output layer. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss.
Plot Validation Accuracy Function: A function is defined to plot validation accuracy over epochs.
Training the Baseline Model: 
The baseline CNN model is trained on the CIFAR-10 training data for 100 epochs with a batch size of 32. Validation data is used for validation during training.

Data Augmentation: 
Various data augmentation techniques such as brightening, darkening, and horizontal flipping are applied to the training data. Augmented data is stored and concatenated in smaller batches.
Building and Training Augmented Models: 
Multiple CNN models are trained using the augmented data. Each model is trained for 100 epochs with a batch size of 32.
Plotting Validation Accuracy: Validation accuracy of both the baseline and augmented models is plotted over epochs to compare their performance.

Conclusion:
Data augmentation is a powerful technique to improve model generalization by increasing the diversity of the training data. By applying various transformations to the original images, augmented data can help CNN models learn robust features and improve their performance on unseen data. This code snippet serves as an example of implementing data augmentation for image classification tasks using TensorFlow and Keras.

# Problem 4
# Self-Supervised Learning via Pretext Tasks

Overview:
This code snippet illustrates the implementation of self-supervised learning via pretext tasks using the CIFAR-10 dataset. Self-supervised learning involves training a model to predict certain properties of the data without manual annotation. In this case, two pretext tasks are employed: vertical flipping and 90-degree rotation. The model is first trained on these pretext tasks and then fine-tuned on the main classification task.

Code Structure:
Data Loading and Preprocessing: CIFAR-10 dataset is loaded and split into baseline and pretext datasets. Pretext datasets are augmented by applying vertical flipping and 90-degree rotation.

Pretext Model Training: 
A CNN model is constructed and trained on the pretext tasks using the augmented pretext dataset. The model architecture includes convolutional layers, max-pooling layers, flatten layer, and fully connected layers with ReLU activation.
Fine-Tuning with Transfer Learning: Weights from the pretext model are transferred to the main model. The last layer of the main model is reinitialized, and a multi-optimizer is defined for fine-tuning. The main model is compiled with an appropriate loss function and metrics.

Training the Main Model: 
The main model is trained on the baseline dataset with the transferred weights and fine-tuned on the main classification task.

Evaluation and Comparison: 
Training history of both the baseline and main models is plotted to compare their performance. Validation accuracy and loss curves are visualized to analyze model performance.

Conclusion:
Self-supervised learning via pretext tasks offers a promising approach to leverage unlabeled data for model training. By pretraining on pretext tasks, the model can learn useful representations of the data, which can then be fine-tuned for downstream tasks. However, fine-tuning performance may vary depending on factors like architecture, hyperparameters, and dataset characteristics. Further improvements can be explored through hyperparameter tuning, regularization techniques, or ensemble methods.
