# Audio-Classification

This repository contains a machine learning model designed to detect the presence of Capuchin bird calls in audio samples recorded from a forest. The model uses a sequential neural network architecture. The project includes a Jupyter Notebook (AudioClassification.ipynb) that covers the entire process of data preprocessing, model training, evaluation, and testing.

# Installation
To run the code in this repository, you need to have Python and Jupyter Notebook installed. The required dependencies are listed in the requirements.txt file. You can install them using pip:

pip install -r requirements.txt


# Usage
The main component of this repository is the Jupyter Notebook:

AudioClassification.ipynb: Jupyter notebook for training the audio classification model to detect Capuchin bird calls.
Running the Notebook
1. Open the Notebook:
   Open AudioClassification.ipynb in Jupyter Notebook.

2. Follow the Instructions:
   Follow the instructions provided in the notebook to preprocess the data, build the model, train it, and evaluate its performance.

# Data Preprocessing
The AudioClassification.ipynb notebook includes a preprocessing function to prepare the audio data for model training. Here's an explanation of the preprocessing steps:

Load WAV File: The preprocessing function begins by loading an audio file. The audio is converted to a single-channel (mono) format and resampled to 16 kHz. This standardizes the audio files, ensuring they have the same sample rate and channel configuration.

Truncate or Pad Audio: The audio is then truncated to a maximum length of 3 seconds (48000 samples at 16 kHz). If the audio is shorter than 3 seconds, it is padded with zeros to reach the required length. This step ensures that all audio inputs to the model have the same length, simplifying the processing and model training.

Zero Padding: If the audio file is shorter than 3 seconds, the function adds zero-padding to the end of the audio signal to make up the difference. This guarantees a consistent input size for the neural network.

STFT (Short-Time Fourier Transform): The preprocessed audio is then transformed into a spectrogram using STFT. The STFT converts the time-domain signal into a time-frequency representation, which helps in analyzing the frequency components of the audio signal over time.

Magnitude Spectrogram: The magnitude of the complex STFT result is computed to create a spectrogram that represents the intensity of frequencies over time. This spectrogram is a crucial input feature for the neural network, as it captures the essential characteristics of the audio signal.

Expand Dimensions: Finally, the spectrogram is reshaped to add a channel dimension, making it compatible with the input requirements of convolutional neural networks (CNNs). This step prepares the spectrogram for input into the sequential model.

# Model Training
The AudioClassification.ipynb notebook covers the following steps:

1. Data Loading: Load the dataset containing audio samples labeled for the presence of Capuchin bird calls.
2. Data Preprocessing: Preprocess the audio samples using the function described above.
3. Model Definition: Define a sequential neural network architecture using TensorFlow and Keras.
4. Model Training: Train the model using the preprocessed data.
5. Model Evaluation: Evaluate the model's performance on a validation set and visualize the results.




This README file provides an overview of the repository, including instructions for installation, usage, data preprocessing, model training, and contributing. If you have any questions or need further assistance, feel free to open an issue in the repository.





