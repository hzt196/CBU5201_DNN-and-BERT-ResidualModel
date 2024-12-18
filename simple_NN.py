import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split as spl
from sklearn.neighbors import KNeighborsClassifier as mod
from sklearn.metrics import roc_curve as rcr, auc as aucm, classification_report as rpt, accuracy_score as scr, confusion_matrix as cmx
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler as ovr

aud_path = "Deception-main\story_audiofiles"
csv_path = "E:\PythonProject_test\Deception-main\MLEndDeception_small_sample.csv"

dataset = pd.read_csv(csv_path)
dataset['cls'] = dataset['label'].map({"deceptive_story": 0, "true_story": 1})

def extract(fil):
    snd, rate = librosa.load(fil, duration=30, sr=None)
    coeff = librosa.feature.mfcc(y=snd, sr=rate, n_mfcc=40)
    return np.mean(coeff.T, axis=0)

features = []
classes = []

for _, row in dataset.iterrows():
    filepath = os.path.join(aud_path, row['filename'])
    feat = extract(filepath)
    if feat is not None:
        features.append(feat)
        classes.append(row['cls'])

X_arr = np.array(features)
y_arr = np.array(classes)

augmenter = ovr(
    sampling_strategy=lambda y: {
        cls: count * 2 if count == min(np.bincount(y)) else count
        for cls, count in zip(*np.unique(y, return_counts=True))
    },
    random_state=42
)
X_arr, y_arr = augmenter.fit_resample(X_arr, y_arr)

X_train, X_val, y_train, y_val = spl(X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr)

model = mod(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(rpt(y_val, y_pred))
print("Accuracy:", scr(y_val, y_pred))
