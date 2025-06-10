import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# === 1. CARGAR TU CSV ===
df = pd.read_csv('C:/Users/ASUS/Documents/Semestre05/ESTADISTICA COMPUTACIONAL/predicion/abb.csv')
 # Cambia la ruta si hace falta

# --- 2. Crear columna fecha ---
df['fecha'] = pd.to_datetime(df['ano'].astype(str) + '-W' + df['semana'].astype(str) + '-1', format='%Y-W%W-%w')

# --- 3. Agrupar por fecha (contar casos) ---
serie = df.groupby('fecha').size()

# --- 4. Escalar datos ---
scaler = MinMaxScaler(feature_range=(0,1))
serie_scaled = scaler.fit_transform(serie.values.reshape(-1,1))

# --- 5. Crear secuencias para LSTM ---
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # semanas para predecir la siguiente
X, y = create_sequences(serie_scaled, seq_length)

# --- 6. Dividir en train y test ---
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 7. Crear modelo LSTM ---
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# --- 8. Entrenar ---
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# --- 9. Predecir ---
y_pred = model.predict(X_test)

# --- 10. Invertir escala ---
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# --- 11. Calcular RMSE ---
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f'RMSE LSTM: {rmse:.2f}')

# --- 12. Graficar resultados (predicción en test) ---
plt.figure(figsize=(10,5))
plt.plot(serie.index[-len(y_test):], y_test_inv, label='Real')
plt.plot(serie.index[-len(y_test):], y_pred_inv, label='Predicción')
plt.title('Predicción LSTM de casos de dengue (test set)')
plt.xlabel('Fecha')
plt.ylabel('Casos')
plt.legend()
plt.show()

# --- 13. Predicción hacia el futuro hasta el 31-dic-2025 ---

last_date = serie.index[-1]
end_date = pd.to_datetime('2025-12-31')

# Calcular número de semanas desde last_date hasta end_date
future_weeks = ((end_date - last_date).days // 7) + 1

# Tomar últimas seq_length semanas escaladas para iniciar predicción
last_sequence = serie_scaled[-seq_length:].reshape(1, seq_length, 1)

future_predictions = []

for _ in range(future_weeks):
    next_pred = model.predict(last_sequence)[0][0]
    future_predictions.append(next_pred)
    last_sequence = np.append(last_sequence[:,1:,:], [[[next_pred]]], axis=1)

# Invertir escala para valores reales
future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

# Crear fechas futuras desde la semana siguiente a last_date hasta end_date
future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_weeks, freq='W-MON')

# Graficar datos reales + predicción futura hasta 2025
plt.figure(figsize=(12,6))
plt.plot(serie.index, serie.values, label='Datos reales')
plt.plot(future_dates, future_predictions_inv, label=f'Predicción hasta 2025', color='red')
plt.title('Predicción LSTM de casos de dengue - Hasta 2025')
plt.xlabel('Fecha')
plt.ylabel('Casos')
plt.legend()
plt.show()
