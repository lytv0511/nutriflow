import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, Input
import os

# ==== 0. Generate synthetic data and save as CSV if not present ====

csv_path = "health_data.csv"
if not os.path.exists(csv_path):
    num_days = 120
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days)
    df_synth = pd.DataFrame({
        'date': dates,
        'HRV': np.random.normal(50, 10, num_days),
        'RHR': np.random.normal(60, 5, num_days),
        'Sleep': np.random.normal(7, 1, num_days),
        'Steps': np.random.normal(8000, 2000, num_days),
        'Mood': np.random.normal(3, 1, num_days),
        'Readiness': np.random.normal(75, 10, num_days)
    })
    df_synth.to_csv(csv_path, index=False)
    print(f"Synthetic data generated and saved to {csv_path}")

# ==== 1. Load and preprocess data ====

df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.sort_values('date')

feature_cols = ['HRV', 'RHR', 'Sleep', 'Steps', 'Mood']
target_col = 'Readiness'

df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

# ==== 2. Create sequences for model input ====

sequence_length = 7

X = []
y = []

for i in range(len(df) - sequence_length):
    seq = df[feature_cols].iloc[i:i+sequence_length].values
    label = df[target_col].iloc[i+sequence_length]
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Shape of input X: {X.shape}")
print(f"Shape of labels y: {y.shape}")

# ==== 3. Build the Conv1D model (TFLite-compatible) ====

def build_readiness_model(seq_len, num_features):
    model = Sequential([
        Input(shape=(seq_len, num_features), name="input"),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_readiness_model(sequence_length, len(feature_cols))
model.summary()

# ==== 4. Train the model ====

history = model.fit(
    X, y,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    verbose=2
)

# ==== 5. Export to TensorFlow Lite model ====

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("readiness_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as readiness_model.tflite")
