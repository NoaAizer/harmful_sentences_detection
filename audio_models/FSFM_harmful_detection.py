"""
Audio Model.
Extract five sound features for each of the audio samples and apply the FSFM model.

Results represented on 5-fold.

Source: https://github.com/neiterman21/cheat_detector_keras
"""


import os
import librosa
import ast

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dropout, Dense
from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight

from datetime import datetime


# Define labels
# 0 = neutral speech
# 2 = insulting speech
# 4 = unsafe speech

# if both False = (0,4) classes
second = True  # Two classes (0,2)
three = False  # Three classes (0,2,4)

create_features = False  # On the first time of running the model on the data
num_of_steps = 100

# ## GLOBAL SEED ##
SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.random.set_seed(SEED)
np.random.seed(SEED)

if second:
    csv_path = 'merav2-all.csv'
    save_path = 'merav2_with_fsfm.csv'
elif three:
    csv_path = 'merav3-all.csv'
    save_path = 'merav3_with_fsfm.csv'

else:
    csv_path = 'merav_all.csv'
    save_path = 'merav_with_fsfm.csv'


def NN_model():
    """
    FSFM model structure
    """
    model = Sequential()

    model.add(Dense(193, input_shape=(193,), activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    if three:
        model.add(Dense(3, activation='softmax'))
    else:
        model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam', )

    return model


def get_results(model, X, Y, test):
    print('Confusion Matrix')
    Y = np.argmax(Y, axis=1)
    t = np.asarray(X)[test]
    l = np.asarray(Y)[test]
    y_pred = model.predict(t)
    y_pred = np.argmax(y_pred, axis=1)
    if three:
        labels = [0, 1, 2]
    else:
        labels = [0, 1]

    cm = confusion_matrix(l, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    if three:
        print(classification_report(l, y_pred))
    else:
        Y_test = np.asarray(Y)[test]
        from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
        print(f"Accuracy: {round(accuracy_score(Y_test, y_pred), 2)}")
        print(f"Precision: {round(precision_score(Y_test, y_pred), 2)}")
        print(f"Recall: {round(recall_score(Y_test, y_pred), 2)}")
        print(f"F1_score: {round(f1_score(Y_test, y_pred), 2)}\n\n")
    return cm


def read_fsfm_col(df):
    """
    Convert the five sound features column from string to numpy array.
    """
    vecs = []
    for row in df.iterrows():
        # Convert the vector to list (from string representation)
        res = ast.literal_eval(row[1]['fsfm'])
        vecs.append(np.array(res))
    return np.array(vecs)


def extract_features(files):
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(str(files.file))

    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = 0
    try:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                                  sr=sample_rate).T, axis=0)
    except:
        print(file_name)

    # We add also the classes of each file as a label at the end
    label = files.label

    return mfccs, chroma, mel, contrast, tonnetz, label


def nn_preprocess_step(df, name=""):
    startTime = datetime.now()
    print("Starting featureextraction at: ", startTime)
    features_label = df.apply(extract_features, axis=1)
    print("done extracting. took: ", datetime.now() - startTime)

    # Saving the numpy array because it takes a long time to extract the features
    np.save(name, features_label)

    # loading the features
    features_label = np.load(name + '.npy', allow_pickle=True)
    # We create an empty list where we will concatenate all the features into one long feature
    # for each file to feed into our neural network
    features = []
    labels = []
    for i in range(0, len(features_label)):
        try:
            features.append(np.concatenate((features_label[i][0], features_label[i][1],
                                            features_label[i][2], features_label[i][3],
                                            features_label[i][4]), axis=0))
            labels.append(features_label[i][5])
        except:
            print("feature ", i, "didnt work")

    np.unique(labels, return_counts=True)

    # Setting our X as a numpy array to feed into the neural network
    X = np.array(features)
    # Setting our y
    y = np.array(labels)

    # Hot encoding y
    lb = LabelEncoder()
    if three:
        y = to_categorical(lb.fit_transform(y))
    else:
        y = to_categorical(1 - lb.fit_transform(y))

    return X, y


def nn_preprocess(train, val):
    X_train, y_train = nn_preprocess_step(train, "train_features")
    X_val, y_val = nn_preprocess_step(val, "val_features")

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_val)

    return X_train, y_train, X_val, y_val

def main():
    print("---------------------  FSFM ---------------------")
    if create_features:
        df = pd.read_csv(csv_path, delimiter=',')
        df.rename(columns={'path': 'file'}, inplace=True)
        # Extract features
        X, Y = nn_preprocess_step(df, "test_features")
        ss = StandardScaler()
        X = ss.fit_transform(X)

        # Save a column of five sound features in a new csv file
        df['fsfm'] = X.tolist()
        df.to_csv(save_path)

    else:
        df = pd.read_csv(save_path, delimiter=',')
        df.rename(columns={'path': 'file'}, inplace=True)

        X = read_fsfm_col(df)
        # Setting y
        y = np.array(df.label)
        # Hot encoding y
        lb = LabelEncoder()
        if three:
            Y = to_categorical(lb.fit_transform(y))
        else:
            Y = to_categorical(1 - lb.fit_transform(y))

    # define 5-fold cross validation test harness
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cvscores = []
    if three:
        models = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        all_l = []
        all_pred = []

    else:
        models = [[0, 0], [0, 0]]
    first = True

    for train, test in kfold.split(X, np.argmax(Y, axis=1)):
        print(*train, sep=",")
        print(*test, sep=",")
        model = NN_model()

        # weighted loss
        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(df.label),
                                                    y=df.label[train])
        weights = {i: weights[i] for i in range(len(np.unique(df.label)))}
        print(weights)
        print(len(X))
        loss, acc = model.evaluate(X[test], Y[test], verbose=0)
        print("before fit: loss = ", loss, " accuracy=", acc)

        for i in range(num_of_steps):
            history = model.fit(X[train], Y[train], verbose=0, class_weight=weights)
            print(i, "acc: ", history.history['accuracy'], "loss: ", history.history['loss'])

            # prevent over-fitting
            if float(history.history['accuracy'][0]) > 0.9999:
                break

            # evaluate each 10 training steps
            if i % 10 == 0:
                loss, acc = model.evaluate(X[test], Y[test], verbose=0)
                print(str(i), ": loss = ", loss, " accuracy=", acc)

        # final evaluation
        scores = model.evaluate(X[test], Y[test])
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
        models += get_results(model, X, Y, test)
        y_pred = model.predict(X[test])
        if three:
            if first:
                all_l = np.argmax(Y[test], axis=1)
                all_pred = np.argmax(y_pred, axis=1)
            else:
                all_l = [*all_l, *np.argmax(Y[test], axis=1)]
                all_pred = [*all_pred, *np.argmax(y_pred, axis=1)]
                all_pred = [*all_pred, *np.argmax(y_pred, axis=1)]

        first = False

    print(models)
    if three:
        print(classification_report(all_l, all_pred))
    else:
        TP = models[1][1]
        FP = models[0][1]
        FN = models[1][0]
        TN = models[0][0]
        print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
        print(f"Precision: {(TP) / (TP + FP)}")
        print(f"Recall: {TP / (TP + FN)}")
        print(f"F1_score: {(2 * TP / (TP + FN) * (TP) / (TP + FP)) / (TP / (TP + FN) + (TP) / (TP + FP))}\n\n")
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


if __name__ == "__main__":
    main()
