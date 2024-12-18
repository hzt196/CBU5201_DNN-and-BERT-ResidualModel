import librosa
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

# Set file paths
aud_path = "Deception-main\story_audiofiles"
csv_path = "E:\PythonProject_test\Deception-main\MLEndDeception_small_sample.csv"

# Load the dataset
dataset = pd.read_csv(csv_path)
dataset['cls'] = dataset['label'].map({"deceptive_story": 0, "true_story": 1})

# Feature extraction function
def extract(fil):
    snd, rate = librosa.load(fil, duration=30, sr=None)
    coeff = librosa.feature.mfcc(y=snd, sr=rate, n_mfcc=40)
    return np.mean(coeff.T, axis=0)

# Process audio files to extract features and labels
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

# Handle class imbalance using RandomOverSampler
augmenter = RandomOverSampler(
    sampling_strategy=lambda y: {
        cls: count * 2 if count == min(np.bincount(y)) else count
        for cls, count in zip(*np.unique(y, return_counts=True))
    },
    random_state=42
)
X_arr, y_arr = augmenter.fit_resample(X_arr, y_arr)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr)

# Create an MLP model pipeline
def create_mlp_pipeline(hidden_layers=(100, 50), learning_rate=0.001):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=learning_rate,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ))
    ])

# Train and evaluate the MLP model
def train_and_evaluate_mlp(X_train, X_val, y_train, y_val, hidden_layers=(100, 50), learning_rate=0.001):
    pipeline = create_mlp_pipeline(hidden_layers, learning_rate)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    print(f"Architecture: {hidden_layers}")
    print(f"Learning rate: {learning_rate}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    return pipeline

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Plot learning curves
def plot_learning_curves(model):
    plt.figure(figsize=(10, 6))
    plt.plot(model.named_steps['mlp'].loss_curve_, label='Training Loss')
    if model.named_steps['mlp'].validation_scores_ is not None:
        plt.plot(model.named_steps['mlp'].validation_scores_, label='Validation Score')
    plt.title('Learning Curves')
    plt.xlabel('Iterations')
    plt.ylabel('Loss / Score')
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform grid search for model hyperparameter tuning
def perform_grid_search(X_train, y_train):
    param_grid = {
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50), (100, 50, 25)],
        'mlp__learning_rate_init': [0.01, 0.001, 0.0001],
        'mlp__alpha': [0.0001, 0.001, 0.01]
    }
    pipeline = create_mlp_pipeline()
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print("\nGrid Search Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

# Train basic model
basic_model = train_and_evaluate_mlp(X_train, X_val, y_train, y_val)
plot_confusion_matrix(y_val, basic_model.predict(X_val))
plot_learning_curves(basic_model)

# Perform grid search for optimal parameters
print("\nPerforming grid search for optimal parameters...")
best_model = perform_grid_search(X_train, y_train)
best_model = perform_grid_search(X_train, y_train)
print("\nEvaluating best model from grid search:")
y_pred_best = best_model.predict(X_val)
print(classification_report(y_val, y_pred_best))
plot_confusion_matrix(y_val, y_pred_best)
plot_learning_curves(best_model)
