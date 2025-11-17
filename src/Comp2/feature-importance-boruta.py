import duckdb
import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
import gc 

# --- 1. Configuración de Archivos ---
FEATURES_PARQUET_PATH = 'features_dataset.parquet'
BORUTA_OUTPUT_PATH = 'feature_boruta_selection.csv'

# --- 2. Configuración de Pre-filtrado ---
TOP_N_FEATURES_FROM_LGBM = 1000
IMPORTANCE_INPUT_PATH = 'feature_importances_avg.csv'
# ---------------------------------

# --- 3. Configuración del Split (Time-Series) ---
TRAIN_START_MONTH = 202007
TRAIN_END_MONTH = 202102
# ---------------------------------

print(f"--- Iniciando Selección de Features con Boruta (Top {TOP_N_FEATURES_FROM_LGBM}) ---")

# --- Cargar la lista de features pre-filtradas ---
print(f"Cargando lista de features pre-filtradas desde: {IMPORTANCE_INPUT_PATH}")
if not os.path.exists(IMPORTANCE_INPUT_PATH):
    print(f"Error: No se encontró el archivo de importancias '{IMPORTANCE_INPUT_PATH}'.")
    print("Por favor, ejecuta 'seleccion_features.py' primero.")
    sys.exit(1)

try:
    df_top_features = pd.read_csv(IMPORTANCE_INPUT_PATH)
    top_n_features_list = df_top_features.head(TOP_N_FEATURES_FROM_LGBM)['feature'].tolist()
    print(f"Se usarán las {len(top_n_features_list)} features más importantes del LGBM.")
except Exception as e:
    print(f"Error al leer el archivo CSV de importancias: {e}")
    sys.exit(1)
# ----------------------------------------------------

# --- 4. Definir Columnas Necesarias (ANTES de la consulta) ---
target = 'clase_ternaria'

cols_to_drop = [
    'numero_de_cliente', 'foto_mes', 'clase_ternaria',
    'Master_status', 'Visa_status'
]

features = [col for col in top_n_features_list if col not in cols_to_drop]

cat_features = [
    'anio', 'mes', 'peor_estado_tarjetas', 'active_quarter', 'cliente_vip',
    'internet', 'tcallcenter', 'thomebanking', 'tmobile_app',
    'Master_delinquency', 'Visa_delinquency', 'ccaja_seguridad'
]

cols_to_load = list(set(features + [target, 'foto_mes']))
cols_to_load = list(set(cols_to_load + cat_features + ['cmobile_app_trx_pre202010', 'cmobile_app_trx_post202010']))

try:
    temp_con = duckdb.connect(database=':memory:')
    parquet_cols = temp_con.execute(f"DESCRIBE SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')").fetchdf()['column_name'].tolist()
    temp_con.close()
    
    cols_to_load_final = [col for col in cols_to_load if col in parquet_cols]
    features = [col for col in features if col in parquet_cols] 
    print(f"Columnas a cargar de DuckDB: {len(cols_to_load_final)}")
except Exception as e:
    print(f"Advertencia: No se pudo leer el esquema del Parquet, se intentará cargar todas las {len(cols_to_load)} columnas. Error: {e}")
    cols_to_load_final = cols_to_load

# ----------------------------------------------------

print(f"Cargando dataset: {FEATURES_PARQUET_PATH}")

if not os.path.exists(FEATURES_PARQUET_PATH):
    print(f"Error: No se encontró el archivo de features '{FEATURES_PARQUET_PATH}'.")
    sys.exit(1)

con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB establecida.")

try:
    print("Construyendo consulta SQL optimizada...")
    quoted_cols = [f'"{col}"' for col in cols_to_load_final]
    
    query = f"""
    SELECT {', '.join(quoted_cols)}
    FROM read_parquet('{FEATURES_PARQUET_PATH}')
    WHERE foto_mes >= {TRAIN_START_MONTH} AND foto_mes <= {TRAIN_END_MONTH}
    """
    
    print(f"Ejecutando consulta para cargar datos de entrenamiento ({TRAIN_START_MONTH} a {TRAIN_END_MONTH})...")
    df_train = con.execute(query).fetchdf()
    print(f"Datos de entrenamiento cargados. Forma: {df_train.shape}")
    
    if df_train.empty:
        print("\n¡ADVERTENCIA! El set de entrenamiento está vacío. Revisa la lógica de fechas.")
        sys.exit(1)

    print("Forzando conversión de tipos (casting)...")
    if 'cmobile_app_trx_pre202010' in df_train.columns:
        df_train['cmobile_app_trx_pre202010'] = pd.to_numeric(df_train['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df_train.columns:
        df_train['cmobile_app_trx_post202010'] = pd.to_numeric(df_train['cmobile_app_trx_post202010'], errors='coerce').fillna(0)

    # --- 5. Preparación de Datos para el Modelo ---
    
    df_train[target] = df_train[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
    df_train[target] = df_train[target].fillna(0).astype(int)
    print("\nConteo del nuevo target binario:")
    print(df_train[target].value_counts(normalize=True).to_string())
    
    cat_features_final = []
    for col in cat_features:
        if col in df_train.columns and col in features:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(int)
            cat_features_final.append(col)
            
    print(f"\nTotal de features a analizar en Boruta: {len(features)}")
    
    print("Optimizando Dtypes (float64->float32, int64->int32)...")
    for col in features:
        if col not in cat_features_final: 
            if df_train[col].dtype == 'float64':
                df_train[col] = df_train[col].astype('float32')
            elif df_train[col].dtype == 'int64':
                df_train[col] = df_train[col].astype('int32')

    # --- 6. Crear Splits de Entrenamiento y Liberar Memoria ---
    
    X_train = df_train[features]
    y_train = df_train[target]

    # --- NUEVO: Calcular scale_pos_weight ---
    # Lo calculamos desde el 'y_train' de Pandas ANTES de borrarlo.
    try:
        counts = y_train.value_counts()
        scale_pos_weight = counts[0] / counts[1]
        print(f"Calculando scale_pos_weight para desbalanceo: {scale_pos_weight:.2f}")
    except Exception as e:
        print(f"Advertencia: No se pudo calcular scale_pos_weight. {e}. Usando 1.0.")
        scale_pos_weight = 1.0
    # ----------------------------------------
    
    print(f"Tamaño de Entrenamiento (para Boruta): {X_train.shape[0]} registros")
    
    print("Liberando memoria del DataFrame de entrenamiento...")
    del df_train
    gc.collect()

    # --- 7. Preparar Datos para Boruta ---
    print("Convirtiendo features categóricas a códigos enteros...")
    for col in cat_features_final:
        X_train[col] = X_train[col].astype('category').cat.codes
        
    print("Llenando NaNs/NAs restantes en features numéricas con 0...")
    numeric_features = [col for col in features if col not in cat_features_final]
    X_train.loc[:, numeric_features] = X_train.loc[:, numeric_features].fillna(0)
        
    print("Convirtiendo a arrays de NumPy...")
    X_train_values = X_train.values
    y_train_values = y_train.values
    
    print("Liberando memoria de los DataFrames X/y...")
    del X_train
    del y_train
    gc.collect()
    
    # --- 8. Configurar y Ejecutar Boruta ---

    print("\nIniciando BorutaPy (esto puede tardar varios minutos)...")

    # 1. Configurar el estimador (LightGBM) que usará Boruta
    # --- MODIFICADO ---
    lgb_estimator = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=100,
        learning_rate=0.05,
        n_jobs=4, 
        # seed=42, # Eliminado: Boruta(random_state=42) ya lo maneja
        colsample_bytree=0.7,
        subsample=0.7,
        # Reemplazado 'is_unbalanced=True' (obsoleto) por 'scale_pos_weight'
        scale_pos_weight=scale_pos_weight 
    )

    # 2. Configurar Boruta
    boruta_selector = BorutaPy(
        estimator=lgb_estimator,
        verbose=2,
        random_state=42, # Esto controla la aleatoriedad de Boruta y sus estimadores
        max_iter=100
    )

    # 3. Entrenar Boruta
    boruta_selector.fit(X_train_values, y_train_values)

    print("\nEntrenamiento de Boruta completado.")

    # --- 9. Extraer y Guardar Resultados de Boruta ---
    
    print("Extrayendo resultados de Boruta...")
    
    df_boruta = pd.DataFrame({
        'feature': features,
        'ranking': boruta_selector.ranking_,
        'support_': boruta_selector.support_, 
        'support_weak_': boruta_selector.support_weak_
    })
    
    def get_decision(row):
        if row['support_']:
            return 'Confirmada'
        elif row['support_weak_']:
            return 'Tentativa'
        else:
            return 'Rechazada'
            
    df_boruta['decision'] = df_boruta.apply(get_decision, axis=1)
    
    df_boruta = df_boruta.sort_values(by='ranking', ascending=True).reset_index(drop=True)
    
    df_boruta.to_csv(BORUTA_OUTPUT_PATH, index=False)
    
    print(f"\n--- ¡Selección de Features Boruta Completada! ---")
    print(f"Lista completa (Confirmada/Tentativa/Rechazada) guardada en: {BORUTA_OUTPUT_PATH}")
    
    print("\n--- FEATURES CONFIRMADAS ---")
    confirmed_features = df_boruta[df_boruta['decision'] == 'Confirmada']['feature'].tolist()
    print(confirmed_features)
    print(f"\nTotal confirmadas: {len(confirmed_features)} de {len(features)}")
    print("----------------------------")


except Exception as e:
    print(f"\nHa ocurrido un error durante el proceso de selección: {e}")

finally:
    if 'con' in locals() or 'con' in globals():
        con.close()
        print("\nConexión a DuckDB cerrada.")
