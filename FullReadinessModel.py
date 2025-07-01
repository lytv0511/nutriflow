import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import coremltools as ct

# ==== 0. Generate synthetic data and save as CSV if not present ====
import os

csv_path = "health_data.csv"
if not os.path.exists(csv_path):
    num_days = 120  # or any number you want
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

# CSV must have columns: date, HRV, RHR, Sleep, Steps, Mood, Readiness
df = pd.read_csv(csv_path, parse_dates=['date'])

# Sort by date (important!)
df = df.sort_values('date')

# Features and target
feature_cols = ['HRV', 'RHR', 'Sleep', 'Steps', 'Mood']
target_col = 'Readiness'

# Normalize features (mean=0, std=1)
df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

# ==== 2. Create sequences for LSTM input ====

sequence_length = 7

X = []
y = []

for i in range(len(df) - sequence_length):
    seq = df[feature_cols].iloc[i:i+sequence_length].values
    label = df[target_col].iloc[i+sequence_length]  # next day readiness
    X.append(seq)
    y.append(label)

X = np.array(X)  # shape: (num_samples, seq_len, num_features)
y = np.array(y)  # shape: (num_samples,)

print(f"Shape of input X: {X.shape}")
print(f"Shape of labels y: {y.shape}")

# ==== 3. Build the LSTM model ====

def build_readiness_model(seq_len, num_features):
    model = Sequential([
        Input(shape=(seq_len, num_features), name="input"),  # Explicitly name input
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
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

# ==== 5. Save the model for CoreML conversion later ====

model.save("readiness_lstm_model.keras")
print("Model saved to readiness_lstm_model.keras")

# ==== 6. Convert to CoreML model ====

keras_model = keras.models.load_model("readiness_lstm_model.keras", compile=False)
coreml_model = ct.convert(
    keras_model,
    source="tensorflow",
    inputs=[ct.TensorType(shape=(1, 7, 5), name="input")],  # Match the explicit input name
    convert_to="mlprogram"
)

coreml_model.save("ReadinessModel.mlpackage")
print("CoreML model successfully saved as ReadinessModel.mlpackage")