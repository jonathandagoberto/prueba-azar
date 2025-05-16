# lottery_advanced_ai.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import optuna
import psutil
from sklearn.ensemble import IsolationForest
from keras.layers import LSTM, Dense, Input, Attention, GlobalAveragePooling1D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from torch import nn
from sklearn.model_selection import train_test_split
from numba import cuda

# ----------------------------------------
# CONFIGURACIÓN
# ----------------------------------------
DATA_PATH = 'prueba_estudio.csv'
WINDOW = 15
TEST_SIZE = 0.15
N_CLUSTERS = 6
MIN_NUMBER = 1
MAX_NUMBER = 60
HISTORICAL_WEIGHT = 0.2  # Peso para la media histórica

# ----------------------------------------
# VALIDACIONES INICIALES
# ----------------------------------------
if psutil.virtual_memory().available < 2 * 1024**3:
    raise MemoryError("Memoria insuficiente para ejecutar el modelo")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el archivo: {DATA_PATH}")

# ----------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ----------------------------------------
try:
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("El archivo CSV está vacío.")
    
    df.columns = df.columns.str.lower().str.replace(r'[\W_]+', '', regex=True)
    required_cols = [f'num{i}' for i in range(1,7)]
    df = df[required_cols]
    
    nums = df.values.astype(np.float32)
    if np.any((nums < MIN_NUMBER) | (nums > MAX_NUMBER)):
        print("Ajustando números fuera de rango...")
        nums = np.clip(nums, MIN_NUMBER, MAX_NUMBER)

except KeyError as e:
    raise ValueError(f"Error en columnas: {e}") from e

# ----------------------------------------
# LIMPIEZA DE DATOS
# ----------------------------------------
iso = IsolationForest(contamination=0.01, random_state=42)
clean_mask = iso.fit_predict(nums) == 1
nums_clean = nums[clean_mask]

if len(nums_clean) < WINDOW * 2:
    raise ValueError(f"Datos insuficientes: {len(nums_clean)} muestras")

# Calcular media histórica
historical_mean = np.mean(nums_clean, axis=0)

# ----------------------------------------
# ESCALADO CUÁNTICO
# ----------------------------------------
class RobustQuantumScaler:
    def __init__(self):
        self.theta = None
        self.epsilon = 1e-8
    
    def fit_transform(self, X):
        X_float = X.astype(np.complex128)
        fft_result = np.fft.fft(X_float, axis=0)
        self.theta = np.angle(fft_result)
        return np.abs(fft_result) + self.epsilon
    
    def inverse_transform(self, X):
        return np.real(np.fft.ifft((X - self.epsilon) * np.exp(1j * self.theta)))

scaler = RobustQuantumScaler()
nums_scaled = scaler.fit_transform(nums_clean)

# ----------------------------------------
# MODELO HÍBRIDO MEJORADO
# ----------------------------------------
def build_hybrid_model(input_shape, lr=1e-3, units=128):
    inputs = Input(shape=input_shape)
    x = LSTM(units, return_sequences=True)(inputs)  # Más unidades LSTM
    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)  # Capa más ancha
    x = Dropout(0.3)(x)  # Regularización
    outputs = Dense(nums_clean.shape[1], activation='linear')(x)
    
    model = Model(inputs, outputs)
    model.compile(loss='huber', optimizer=Adam(learning_rate=lr), metrics=['mae'])  # Pérdida Huber
    return model

# ----------------------------------------
# GAN MEJORADO
# ----------------------------------------
class EnhancedLotteryGAN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(64, 256), 
            nn.LeakyReLU(0.2), 
            nn.BatchNorm1d(256),
            nn.Linear(256, 512), 
            nn.LeakyReLU(0.2), 
            nn.BatchNorm1d(512),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, z): 
        return self.generator(z)

gan = EnhancedLotteryGAN(nums_clean.shape[1])

# ----------------------------------------
# PREPARACIÓN DE DATOS
# ----------------------------------------
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(nums_scaled, WINDOW)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, shuffle=False)

# ----------------------------------------
# OPTIMIZACIÓN CON OPTUNA MEJORADA
# ----------------------------------------
def run_optuna_optimization():
    def objective(trial):
        params = {
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'units': trial.suggest_categorical('units', [64, 128, 256])
        }
        
        model = build_hybrid_model(
            (WINDOW, nums_clean.shape[1]), 
            lr=params['lr'],
            units=params['units']
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=params['batch_size'],
            verbose=0
        )
        
        trial.set_user_attr('weights', model.get_weights())
        return model.evaluate(X_val, y_val, verbose=0)[0]

    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )
    study.optimize(objective, n_trials=5)
    return study

# ----------------------------------------
# ENSAMBLE DINÁMICO MEJORADO
# ----------------------------------------
class AdvancedDynamicEnsemble:
    def __init__(self, keras_model, gan_model):
        self.keras_model = keras_model
        self.gan_model = gan_model
        self.gan_model.eval()
    
    def predict(self, keras_input):
        keras_pred = self.keras_model.predict(keras_input, verbose=0)[0]
        with torch.no_grad():
            gan_pred = self.gan_model(torch.randn(1, 64)).numpy()[0]
        # Mezcla con distribución gaussiana
        return 0.6 * keras_pred + 0.3 * gan_pred + 0.1 * np.random.normal(size=keras_pred.shape)

# ----------------------------------------
# POST-PROCESAMIENTO MEJORADO
# ----------------------------------------
def postprocess(prediction):
    try:
        scaled = scaler.inverse_transform(prediction.reshape(1, -1))[0]
        
        # Combinar con media histórica
        blended = scaled * (1 - HISTORICAL_WEIGHT) + historical_mean * HISTORICAL_WEIGHT
        processed = np.round(blended).astype(int)
        
        processed = np.clip(processed, MIN_NUMBER, MAX_NUMBER)
        
        # Generar 6 números únicos con prioridad en los más frecuentes
        unique, counts = np.unique(processed, return_counts=True)
        freq_sorted = [x for x, _ in sorted(zip(unique, counts), key=lambda x: -x[1])]
        
        # Seleccionar y complementar
        final = []
        for num in freq_sorted + list(range(MIN_NUMBER, MAX_NUMBER+1)):
            if len(final) >= 6:
                break
            if num not in final and MIN_NUMBER <= num <= MAX_NUMBER:
                final.append(num)
                
        return np.array(final[:6])
    
    except Exception as e:
        print(f"Error en post-procesamiento: {e}")
        return np.random.choice(range(MIN_NUMBER, MAX_NUMBER+1), 6, replace=False)

# ----------------------------------------
# EJECUCIÓN PRINCIPAL
# ----------------------------------------
def cleanup_resources():
    try:
        tf.keras.backend.clear_session()
        cuda.select_device(0)
        cuda.close()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error limpiando recursos: {e}")

if __name__ == '__main__':
    try:
        print("Optimizando hiperparámetros...")
        study = run_optuna_optimization()
        
        print("\nCargando mejor modelo...")
        best_model = build_hybrid_model(
            (WINDOW, nums_clean.shape[1]),
            lr=study.best_params['lr'],
            units=study.best_params['units']
        )
        best_model.set_weights(study.best_trial.user_attrs['weights'])
        
        print("\nEntrenamiento final...")
        best_model.fit(
            X, y,
            epochs=100,
            batch_size=study.best_params['batch_size'],
            validation_split=0.1,
            verbose=1
        )
        
        print("\nGenerando predicción...")
        last_sequence = nums_scaled[-WINDOW:].reshape(1, WINDOW, -1)
        ensemble = AdvancedDynamicEnsemble(best_model, gan)
        raw_pred = ensemble.predict(last_sequence)
        final_numbers = postprocess(raw_pred)
        
        print(f"\nNúmeros predichos: {np.sort(final_numbers)}")
        print("\nÚltimos números reales:")
        print(nums_clean[-3:])
        print(f"\nMAE del modelo: {study.best_trial.value:.2f}")
        
    except Exception as e:
        print(f"\nError crítico: {str(e)}")
    finally:
        cleanup_resources()