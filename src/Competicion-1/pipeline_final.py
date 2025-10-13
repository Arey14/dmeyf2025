import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
import warnings
import sys

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# FASE 1: CONFIGURACIÓN Y PREPARACIÓN DE DATOS (ESTRUCTURA FINAL)
# -----------------------------------------------------------------------------
print("FASE 1: Configuración y Preparación de Datos")

# --- Constantes de Negocio y Semillas ---
DATA_PATH = './'
DATA_FILE = 'competencia_01.csv' # Asumimos que este archivo ahora contiene hasta 202106
GANANCIA_ACIERTO = 780000
COSTO_ESTIMULO = 20000
SEEDS = [761249, 762001, 763447, 762233, 761807]

# --- Carga de datos usando Pandas ---
try:
    print(f"Cargando el archivo: {DATA_PATH}{DATA_FILE}")
    full_df = pd.read_csv(f"{DATA_PATH}{DATA_FILE}")
except FileNotFoundError:
    print(f"Error: El archivo '{DATA_FILE}' no se encontró en la ruta '{DATA_PATH}'.")
    sys.exit(1)

# --- Separación Temporal de Datos (NUEVA ESTRUCTURA) ---
# Datos para la predicción final de 202106
df_predict = full_df[full_df['foto_mes'] == 202106].copy()

# Datos históricos con target confiable
df_train_optuna = full_df[full_df['foto_mes'].isin([202101, 202102])].copy()
df_val_optuna = full_df[full_df['foto_mes'] == 202103].copy()
df_test = full_df[full_df['foto_mes'] == 202104].copy()
df_train_final = full_df[full_df['foto_mes'].isin([202101, 202102, 202103, 202104])].copy()

print(f"Tamaño del set de entrenamiento (Optuna): {df_train_optuna.shape}")
print(f"Tamaño del set de validación (Optuna): {df_val_optuna.shape}")
print(f"Tamaño del set de testeo: {df_test.shape}")
print(f"Tamaño del set de entrenamiento final: {df_train_final.shape}")
print(f"Tamaño del set para predicción final (input 202106): {df_predict.shape}")

# -----------------------------------------------------------------------------
# FASE 2: INGENIERÍA DE CARACTERÍSTICAS
# -----------------------------------------------------------------------------
print("\nFASE 2: Ingeniería de Características")

def create_features(df):
    """Aplica preprocesamiento y crea nuevas características de negocio."""
    d = df.copy()
    # Esta función ahora maneja el caso donde 'clase_ternaria' no existe (para df_predict)
    if 'clase_ternaria' in d.columns:
        d['target'] = np.where(d['clase_ternaria'] == 'BAJA+2', 1, 0)

    return d

# Aplicar a todos los conjuntos de datos
df_train_optuna_fe = create_features(df_train_optuna)
df_val_optuna_fe = create_features(df_val_optuna)
df_test_fe = create_features(df_test)
df_train_final_fe = create_features(df_train_final)
df_predict_fe = create_features(df_predict)

# Definir variables del modelo
features_to_exclude = ['numero_de_cliente', 'foto_mes', 'clase_ternaria', 'target', 'total_deudas', 'uso_limite_visa', 'uso_limite_master']
features = [col for col in df_train_optuna_fe.columns if col not in features_to_exclude]

X_train_optuna = df_train_optuna_fe[features]
y_train_optuna = df_train_optuna_fe['target']
X_val_optuna = df_val_optuna_fe[features]
y_val_optuna = df_val_optuna_fe['target']
X_test = df_test_fe[features]
y_test = df_test_fe['target']

print(f"Número de características para el modelo: {len(features)}")

# --- Funciones de Cálculo de Ganancia ---
def calculate_max_profit_from_curve(y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    df['ganancia_individual'] = np.where(df['y_true'] == 1, GANANCIA_ACIERTO - COSTO_ESTIMULO, -COSTO_ESTIMULO)
    df['ganancia_acumulada'] = df['ganancia_individual'].cumsum()
    max_ganancia = df['ganancia_acumulada'].max()
    return max_ganancia if max_ganancia > 0 else 0

def find_optimal_n(y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    df['ganancia_individual'] = np.where(df['y_true'] == 1, GANANCIA_ACIERTO - COSTO_ESTIMULO, -COSTO_ESTIMULO)
    df['ganancia_acumulada'] = df['ganancia_individual'].cumsum()
    if df['ganancia_acumulada'].max() > 0:
        return df['ganancia_acumulada'].idxmax() + 1
    return 0

# -----------------------------------------------------------------------------
# FASE 3: OPTIMIZACIÓN CON PROMEDIO DE SEMILLAS POR TRIAL
# -----------------------------------------------------------------------------
print("\nFASE 3: Optimización con Promedio de Semillas por Trial")

def objective(trial):
    undersampling_ratio = trial.suggest_int('undersampling_ratio', 1, 30)
    params = {
        'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 50)
    }
    gains_per_seed = []
    for seed in SEEDS:
        train_data = pd.concat([X_train_optuna, y_train_optuna], axis=1)
        bajas = train_data[train_data['target'] == 1]
        continuas = train_data[train_data['target'] == 0]
        n_continuas_to_keep = int(len(bajas) * undersampling_ratio)
        continuas_undersampled = continuas.sample(n=n_continuas_to_keep, random_state=seed)
        train_undersampled = pd.concat([bajas, continuas_undersampled])
        X_train_us, y_train_us = train_undersampled[features], train_undersampled['target']
        params['random_state'] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_us, y_train_us, eval_set=[(X_val_optuna, y_val_optuna)], callbacks=[lgb.early_stopping(100, verbose=False)])
        val_probs = model.predict_proba(X_val_optuna)[:, 1]
        gains_per_seed.append(calculate_max_profit_from_curve(y_val_optuna.values, val_probs))
    return np.mean(gains_per_seed)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=1800) 
print("\n--- Optimización Robusta Completada ---")
print(f"Mejor ganancia promedio en validación: {study.best_value:,.0f}")
print("Mejores hiperparámetros encontrados:")
print(study.best_params)

# -----------------------------------------------------------------------------
# FASE 4: TESTEO DEL MEJOR MODELO ENCONTRADO
# -----------------------------------------------------------------------------
print("\nFASE 4: Testeo del Mejor Modelo (Train: 01-02, Test: 04)")
best_params = study.best_params
optimal_undersampling_ratio = best_params.pop('undersampling_ratio')
best_model_params = best_params

train_data_test = pd.concat([X_train_optuna, y_train_optuna], axis=1)
bajas_test = train_data_test[train_data_test['target'] == 1]
continuas_test = train_data_test[train_data_test['target'] == 0]
n_continuas_to_keep_test = int(len(bajas_test) * optimal_undersampling_ratio)
continuas_undersampled_test = continuas_test.sample(n=n_continuas_to_keep_test, random_state=SEEDS[0])
train_optimal_us_test = pd.concat([bajas_test, continuas_undersampled_test])
X_train_optimal_us_test, y_train_optimal_us_test = train_optimal_us_test[features], train_optimal_us_test['target']

test_model = lgb.LGBMClassifier(**best_model_params, n_estimators=1000, random_state=SEEDS[0])
test_model.fit(X_train_optimal_us_test, y_train_optimal_us_test)
test_probs = test_model.predict_proba(X_test)[:, 1]
test_ganancia = calculate_max_profit_from_curve(y_test.values, test_probs)
test_auc = roc_auc_score(y_test, test_probs)
optimal_N_for_production = find_optimal_n(y_test.values, test_probs)

print(f"Resultados en el conjunto de Test (202104):")
print(f"  - Ganancia estimada: {test_ganancia:,.0f}")
print(f"  - AUC: {test_auc:.4f}")
print(f"  - Número de envíos óptimo para producción: {optimal_N_for_production}")

# -----------------------------------------------------------------------------
# FASE 5: ENTRENAMIENTO DEL MODELO FINAL Y PREDICCIÓN PARA 202106
# -----------------------------------------------------------------------------
print("\nFASE 5: Entrenamiento del Modelo Final (Train: 01-04) y Predicción para 202106")
final_train_data = pd.concat([df_train_final_fe[features], df_train_final_fe['target']], axis=1)
final_bajas = final_train_data[final_train_data['target'] == 1]
final_continuas = final_train_data[final_train_data['target'] == 0]
final_n_continuas_to_keep = int(len(final_bajas) * optimal_undersampling_ratio)
final_continuas_undersampled = final_continuas.sample(n=final_n_continuas_to_keep, random_state=SEEDS[0])
final_train_undersampled = pd.concat([final_bajas, final_continuas_undersampled])
X_train_final, y_train_final = final_train_undersampled[features], final_train_undersampled['target']

final_model = lgb.LGBMClassifier(**best_model_params, n_estimators=2000, random_state=SEEDS[0])
final_model.fit(X_train_final, y_train_final)
print("Modelo final re-entrenado con todos los datos históricos (submuestreados).")

# Usar los datos de 202106 para predecir 202106
X_predict = df_predict_fe[features]
customer_ids = df_predict_fe['numero_de_cliente']
final_probabilities = final_model.predict_proba(X_predict)[:, 1]

df_final_pred = pd.DataFrame({'numero_de_cliente': customer_ids, 'prob': final_probabilities})
df_final_pred = df_final_pred.sort_values('prob', ascending=False)
clientes_a_contactar = df_final_pred.head(optimal_N_for_production)['numero_de_cliente']
df_final_pred['prediction'] = np.where(df_final_pred['numero_de_cliente'].isin(clientes_a_contactar), 1, 0)

output_filename = 'predicciones_churn_202106.csv'
output_df = df_final_pred[['numero_de_cliente', 'prediction']]
output_df.to_csv(output_filename, index=False)

print(f"\nArchivo de predicciones '{output_filename}' generado con éxito.")
print(f"Se utilizó un N óptimo de: {optimal_N_for_production} envíos.")
print(f"Total de clientes a contactar (prediction=1): {output_df['prediction'].sum()} de {len(output_df)}")
print("\n¡Proceso completado!")
