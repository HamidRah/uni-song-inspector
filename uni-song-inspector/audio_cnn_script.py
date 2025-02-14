# audio_cnn_script.py
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers
import tensorflow as tf
import ast
import librosa
from keras_tuner import RandomSearch, HyperParameters, BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping



train_csv = pd.read_csv('training_data.csv')
test_csv = pd.read_csv('test_data.csv')

labels = train_csv['instrument'].unique()
#spelling error in the test_data we need to account for
label_encoder = {'Sound_Guiatr' : 0}
for index, label in enumerate(labels):
    label_encoder[label] = index

print(labels)
train_csv['instrument'].value_counts()


flat_mel_spec = np.loadtxt(f'./train_mel_spec/00.txt')
mel_spec = flat_mel_spec.reshape(128,44)
plt.figure(figsize=(14,5))

librosa.display.specshow(mel_spec, sr=22050, x_axis='time', y_axis='mel')
plt.title('Piano Mel Spectrogram')
plt.colorbar()


#preparing training data

guitar_df = train_csv[train_csv['instrument'] == 'Sound_Guitar']
piano_df = train_csv[train_csv['instrument'] == 'Sound_Piano']
drum_df = train_csv[train_csv['instrument'] == 'Sound_Drum']
violin_df = train_csv[train_csv['instrument'] == 'Sound_Violin']

# print each instrument dataframe length
print(f'Piano: {len(piano_df)}')
print(f'Guitar: {len(guitar_df)}')
print(f'Drum: {len(drum_df)}')
print(f'Violin: {len(violin_df)}')



#750 examples from each for intail training
data_combind_train_validation = guitar_df['mel spec ref'].tolist()[:1000]
labels_combind_train_validation = guitar_df['instrument'].tolist()[:1000]

data_combind_train_validation.extend(piano_df['mel spec ref'].tolist()[:1000])
labels_combind_train_validation.extend(piano_df['instrument'].tolist()[:1000])

data_combind_train_validation.extend(drum_df['mel spec ref'].tolist()[:1000])
labels_combind_train_validation.extend(drum_df['instrument'].tolist()[:1000])

data_combind_train_validation.extend(violin_df['mel spec ref'].tolist()[:1000])
labels_combind_train_validation.extend(violin_df['instrument'].tolist()[:1000])


#mel spec data and reshape it
for index, data in enumerate(data_combind_train_validation): 
    data_combind_train_validation[index] = np.loadtxt(f'./train_mel_spec/{data}').reshape((128, 44))
 
#change labels to 0-4   
for index, data in enumerate(labels_combind_train_validation):
    labels_combind_train_validation[index] = label_encoder[data]
    


#testing the conversion has worked
plt.figure(figsize=(14,5))
librosa.display.specshow(data_combind_train_validation[0], sr=22050, x_axis='time', y_axis='mel')
plt.title(labels_combind_train_validation[0])
plt.colorbar()


test_csv['instrument'].value_counts()


#spelling mistake in test data - should be guitar not guiatr
guitar_df = test_csv[test_csv['instrument'] == 'Sound_Guiatr']
piano_df = test_csv[test_csv['instrument'] == 'Sound_Drum']
drum_df = test_csv[test_csv['instrument'] == 'Sound_Violin']
violin_df = test_csv[test_csv['instrument'] == 'Sound_Piano']

data_test = guitar_df['mel spec ref'].tolist()
labels_test = guitar_df['instrument'].tolist()

data_test.extend(piano_df['mel spec ref'].tolist())
labels_test.extend(piano_df['instrument'].tolist())

data_test.extend(drum_df['mel spec ref'].tolist())
labels_test.extend(drum_df['instrument'].tolist())

data_test.extend(violin_df['mel spec ref'].tolist())
labels_test.extend(violin_df['instrument'].tolist())

#mel spec data and reshape it
for index, data in enumerate(data_test): 
    data_test[index] = np.loadtxt(f'./test_mel_spec/{data}').reshape((128, 44))
 
#change labels to 0-4   
for index, data in enumerate(labels_test):
    labels_test[index] = label_encoder[data]
    


#history reporter
def report(history, y_pred, y_true):
    plt.plot(history.epoch, history.history["accuracy"], history.history['val_accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.suptitle('Training (blue) and validation (orange) History')
    plt.show()

    # Your label map
    label_map = {'Sound_Guitar': 0, 'Sound_Drum': 1, 'Sound_Violin': 2, 'Sound_Piano': 3}
    # Reverse the label map
    label_map = {v: k for k, v in label_map.items()}

    # Replace numerical labels with instrument names in predictions
    y_pred_names = [label_map[label] for label in np.argmax(y_pred, axis=1)]

    # Replace numerical labels with instrument names in true labels
    y_true_names = [label_map[label] for label in y_true]

    print(f'Test Accuracy {accuracy_score(y_true_names, y_pred_names) * 100}%')

    # Recalculate confusion matrix with instrument names
    cm = confusion_matrix(y_true_names, y_pred_names)
    print(cm)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_map.values(), yticklabels=label_map.values())
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Spectogram CNN')
    plt.show()
    
    #,display_labels=train_csv['instrument'].unique()) 

x_train,x_val,y_train,y_val= train_test_split(data_combind_train_validation, labels_combind_train_validation,
                                                test_size=0.2,
                                                shuffle=True,
                                                stratify=labels_combind_train_validation)

def build_model(hp: HyperParameters):
    input_shape=(128,44,1)
    model = models.Sequential()
    # Adjust precision for Conv2D layers
    model.add(layers.Conv2D(hp.Int('conv_1_units', min_value=32, max_value=128, step=32), (5, 5), activation='relu', input_shape=input_shape, dtype='float32'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # Adjust precision for Conv2D layers
    model.add(layers.Conv2D(hp.Int('conv_2_units', min_value=32, max_value=128, step=32), (5, 5), activation='relu', dtype='float32'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    # Adjust precision for Conv2D layers
    model.add(layers.Conv2D(hp.Int('conv_3_units', min_value=32, max_value=128, step=32), (5, 5), activation='relu', dtype='float32'))
    model.add(layers.Flatten())
    # Adjust precision for Dense layers
    model.add(layers.Dense(hp.Int('dense_1_units', min_value=32, max_value=128, step=32), activation='relu', dtype='float32'))
    model.add(layers.Dropout(0.2))
    # Adjust precision for Dense layers
    model.add(layers.Dense(hp.Int('dense_2_units', min_value=32, max_value=128, step=32), activation='relu', dtype='float32'))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
    return model

tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='bayesian_optimization',
    project_name='audio_cnn'
)

tuner.search_space_summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

tuner.search(np.array(x_train), np.array(y_train),
             epochs=20, 
             validation_data=(np.array(x_val), np.array(y_val)),
             callbacks=[early_stopping])


tuner.results_summary()

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
for hp, value in best_hps.values.items():
    print(f"{hp}: {value}")


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(np.array(x_train), np.array(y_train),
                    epochs=100, 
                    validation_data=(np.array(x_val), np.array(y_val)),
                    callbacks=[early_stopping])

y_pred = model.predict(np.array(data_test))
report(history, y_pred, labels_test)


model.save('./models/instrument_model/1/', save_format='tf')


