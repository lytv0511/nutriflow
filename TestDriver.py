import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, Input, Concatenate
import os

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

df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.sort_values('date')

feature_cols = ['HRV', 'RHR', 'Sleep', 'Steps', 'Mood']
target_col = 'Readiness'

df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

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

def build_hybrid_model(seq_len, num_features):
    input_layer = Input(shape=(seq_len, num_features), name="input")

    conv3 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)   # short-term / ultradian
    conv7 = Conv1D(32, kernel_size=7, activation='relu', padding='same')(input_layer)   # weekly / circadian overlay
    conv14 = Conv1D(32, kernel_size=14, activation='relu', padding='same')(input_layer) # biweekly / infradian

    merged = Concatenate()([conv3, conv7, conv14])
    pooled = GlobalAveragePooling1D()(merged)
    dense = Dense(64, activation='relu')(pooled)
    dropout = Dropout(0.3)(dense)
    output = Dense(1, activation='linear')(dropout)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_hybrid_model(sequence_length, len(feature_cols))
model.summary()

history = model.fit(
    X, y,
    epochs=30,
    batch_size=8,
    validation_split=0.2,
    verbose=2
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("readiness_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as readiness_model.tflite")