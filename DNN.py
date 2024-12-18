import librosa
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split as spl
from sklearn.metrics import classification_report as rpt, confusion_matrix as cmx, roc_curve as rcr, auc as aucm
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
        cls: count * 3 if count == min(np.bincount(y)) else count
        for cls, count in zip(*np.unique(y, return_counts=True))
    },
    random_state=42
)
X_arr, y_arr = augmenter.fit_resample(X_arr, y_arr)

X_train, X_val, y_train, y_val = spl(X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)

y_pred = (model.predict(X_val) > 0.5).astype(int)

print(rpt(y_val, y_pred))
conf_matrix = cmx(y_val, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['Deceptive', 'True'], yticklabels=['Deceptive', 'True'], annot_kws={"fontsize": 12})
plt.title("Confusion Matrix", fontsize=16)
plt.ylabel("Actual", fontsize=14)
plt.xlabel("Predicted", fontsize=14)
plt.show()

fpr, tpr, _ = rcr(y_val, model.predict(X_val).ravel())
roc_auc = aucm(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label="Random Guess")
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curve", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
