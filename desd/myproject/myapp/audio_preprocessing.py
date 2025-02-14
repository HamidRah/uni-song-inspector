import librosa
import numpy as np
import requests
import json

def get_windows(audio, window_size=22050):
    start = 0
    windows = []
    audio_len = len(audio)
    while start < audio_len:
        if start + window_size > audio_len:
            break
        window_end = int(start + window_size)
        windows.append(audio[start:window_end])
        start += int(window_size / 2)
    return windows

def preprocess_audio_for_inference(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    windows = get_windows(audio)
    preprocessed_windows = []
    for window in windows:
        mel = librosa.feature.melspectrogram(y=window, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Ensure the shape matches your model's expected input
        mel_db_resized = np.resize(mel_db, (128, 44))
        mel_db_resized = np.expand_dims(mel_db_resized, axis=-1)  # Adding the channel dimension
        preprocessed_windows.append(mel_db_resized)
    return preprocessed_windows

# # Preprocess your audio file
# audio_path = './static/src/media/80_Gm_LofiPiano_02_823.wav'  # Update this path
# preprocessed_data = preprocess_audio_for_inference(audio_path)

# # print(f"Number of windows: {len(preprocessed_data)}")
# # print(f"Value array: {preprocessed_data[0]}")

# # Write preprocessed data values to a text file
# # with open('G53_data.txt', 'w') as file:
# #     for window in preprocessed_data:
# #         for value in window.flatten():
# #             file.write(str(value) + '\n')
# # print("Preprocessed data values written to preprocessed_data.txt")


# # TensorFlow Serving URL
# url = 'http://localhost:8501/v1/models/instrument_model/versions/2:predict'

# # Prepare data for TensorFlow Serving
# data = json.dumps({"signature_name": "serving_default", "instances": [window.tolist() for window in preprocessed_data]})

# # Send request
# headers = {"Content-Type": "application/json"}
# response = requests.post(url, data=data, headers=headers)

# # Process response
# if response.status_code == 200:
#     predictions = response.json()['predictions']
#     # Process your predictions as needed
#     print(predictions)
# else:
#     print(f"Failed to get predictions, status code: {response.status_code}, response text: {response.text}")

