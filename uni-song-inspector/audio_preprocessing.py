# audio_preprocessing.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers
import tensorflow as tf
from ipywidgets import IntProgress
from IPython.display import display
import librosa
import tqdm
import time 
import os

train_csv = pd.read_csv('./full_data/Metadata_Train.csv')
test_csv = pd.read_csv('./full_data/Metadata_Test.csv')

def get_windows(audio, window_size = 22050):
    
    start = 0
    windows = []
    audio_len = len(audio)
    
    while start < audio_len:
        
        #find window end
        if start+window_size > audio_len:
            break
        else: 
            window_end = int(start + window_size)
        #take window 
        windows.append(audio[start:window_end])
        #move window
        start += int(window_size / 2) 
      
    #stretch any windows the wrong size
    #window stretching appears not to work for now just discard any windows that are too small
    # for index, window in enumerate(windows):
    #     if  len(window) != window_size:
    #         rate = 1/(window_size / len(window))
    #         windows[index] = librosa.effects.time_stretch(y = window, rate=rate)
        
        
    return windows

training_data = {'mel spec ref' : [], 'instrument' : []}

for index, row in tqdm.tqdm_notebook(train_csv.iterrows(), desc = 'tqdm() Progress Bar', total = len(train_csv)):
    
    filename = row['FileName']
    audio, sr = librosa.load(path = f'./full_data/Train_submission/Train_submission/{filename}')
    windowed_audio = get_windows(audio)
    
    for index_au, audio_window in enumerate(windowed_audio):
        
        mel = librosa.feature.melspectrogram(y=audio_window, sr=sr)
        mel_to_db = librosa.power_to_db(mel, ref=np.max)
        flat_mel_spec = mel_to_db.flatten()
        
        filename = f'{index}{index_au}.txt'
        np.savetxt(f'./full_data/train_mel_spec_full/{filename}', flat_mel_spec)
        
        training_data['mel spec ref'].append(filename)
        training_data['instrument'].append(row['Class'])
        
df = pd.DataFrame.from_dict(training_data)
df.to_csv('./full_data/training_data_full.csv')
    

test_data = {'mel spec ref' : [], 'instrument' : []}

for index, row in tqdm.tqdm_notebook(test_csv.iterrows(), desc = 'tqdm() Progress Bar', total = len(test_csv)):
    
    filename = row['FileName']
    audio, sr = librosa.load(path = f'./full_data/Test_submission/Test_submission/{filename}')
    windowed_audio = get_windows(audio, sr)
    
    for index_au, audio_window in enumerate(windowed_audio):
        
        mel = librosa.feature.melspectrogram(y=audio_window, sr=sr)
        mel_to_db = librosa.power_to_db(mel, ref=np.max)
        flat_mel_spec = mel_to_db.flatten()
        
        filename = f'{index}{index_au}.txt'
        np.savetxt(f'./full_data/test_mel_spec_full/{filename}', flat_mel_spec)
        
        test_data['mel spec ref'].append(filename)
        test_data['instrument'].append(row['Class'])
        
     
df = pd.DataFrame.from_dict(test_data)
df.to_csv('./full_data/test_data.csv')
    




