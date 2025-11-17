import duckdb
import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np 
from sklearn.model_selection import train_test_split

# --- 1. Definición de Ganancia ---
PROFIT_TP = 780000  # Ganancia por Verdadero Positivo (detectar un BAJA+2)
COST_FP = -20000    # Costo por Falso Positivo (molestar a un cliente CONTINUA)
# ---------------------------------

# --- 2. Métrica de Ganancia Personalizada ---
def lgbm_profit_metric(y_true, y_pred_probs):
    """
    Métrica de ganancia personalizada para LightGBM.
    Prueba 50 umbrales (thresholds) y devuelve la ganancia máxima.
    """
    # Convertir a arrays de numpy
    y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    
    max_profit = -np.inf  # Empezamos con la peor ganancia posible
    
    # Probar umbrales de 0.01 a 0.50
    thresholds = np.linspace(0.01, 0.50, 50)
    
    for thresh in thresholds:
        # Convertir probabilidades en predicciones (0 o 1)
        y_pred = (y_pred_probs > thresh).astype(int)
        
        # Calcular Verdaderos Positivos (TP) y Falsos Positivos (FP)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Calcular la ganancia total para este umbral
        total_profit = (tp * PROFIT_TP) + (fp * COST_FP)
        
        if total_profit > max_profit:
            max_profit = total_profit
            
    # LightGBM maximiza esta métrica
    # (Nombre de la métrica, valor, ¿es más alto mejor?)
    return 'max_profit', max_profit, True
# ---------------------------------


# --- 3. Configuración de Archivos ---
FEATURES_PARQUET_PATH = 'dataset_202007_202106.parquet'
IMPORTANCE_OUTPUT_PATH = 'feature_importances_avg.csv' # <-- Nuevo nombre

# --- 4. Configuración del Split (Time-Series) ---
TEST_MONTHS = [202104, 202106]
# (NUEVO) Definir el bloque de entrenamiento explícito
TRAIN_START_MONTH = 202007
TRAIN_END_MONTH = 202102
# (TRAIN_EXCLUDE_MONTHS ya no es necesaria)

# --- ¡NUEVO! Configuración de Aleatoriedad ---
# Ejecutaremos el modelo 5 veces con distintas semillas
N_SEEDS = [761249, 762001, 763447, 762233, 761807,800003, 800021, 800051, 800093, 800123,100057, 100207, 100237, 100267, 100297,254881, 499987, 749987, 900007, 378553]
# ---------------------------------

print(f"--- Iniciando Selección de Features (Promediando {len(N_SEEDS)} Seeds) ---")
print(f"Cargando dataset: {FEATURES_PARQUET_PATH}")

if not os.path.exists(FEATURES_PARQUET_PATH):
    print(f"Error: No se encontró el archivo de features '{FEATURES_PARQUET_PATH}'.")
    print("Asegúrate de que 'feature_engineering.py' se haya ejecutado exitosamente.")
    sys.exit(1)

# Conectar a DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB establecida.")

try:
    # Cargar el dataset de features
    df = con.execute(f"SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')").fetchdf()
    print(f"Dataset de features cargado. Forma: {df.shape}")

    # --- Corrección de Dtypes ---
    print("Forzando conversión de tipos (casting)...")
    if 'cmobile_app_trx_pre202010' in df.columns:
        df['cmobile_app_trx_pre202010'] = pd.to_numeric(df['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df.columns:
        df['cmobile_app_trx_post202010'] = pd.to_numeric(df['cmobile_app_trx_post202010'], errors='coerce').fillna(0)


    # --- 5. Preparación de Datos para el Modelo ---
    
    # A. Codificar el Target (BINARIO)
    target = 'clase_ternaria'
    df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
    df[target] = df[target].fillna(0).astype(int)
    print("\nConteo del nuevo target binario:")
    print(df[target].value_counts(normalize=True).to_string())
    
    # B. Definir Features (X) y Target (y)
    cols_to_drop = [
        'numero_de_cliente', 'foto_mes', 'clase_ternaria',
        'Master_status', 'Visa_status' 
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    features = [col for col in df.columns if col not in cols_to_drop]
    
    # C. Convertir columnas categóricas a tipo 'category' para LightGBM
    cat_features = [
        'anio', 'mes', 'peor_estado_tarjetas', 'active_quarter', 'cliente_vip',
        'internet', 'tcallcenter', 'thomebanking', 'tmobile_app', 
        'Master_delinquency', 'Visa_delinquency', 'ccaja_seguridad'
    ]
    
    cat_features_final = []
    for col in cat_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].astype('category')
            cat_features_final.append(col)
        
    print(f"\nTotal de features a entrenar: {len(features)}")
    
    # --- 6. Crear Splits de Entrenamiento y Validación (Time-Series) ---
    
    print(f"Creando split:")
    print(f"  Test (Validación) Meses: {TEST_MONTHS}")
    # --- MODIFICADO ---
    print(f"  Entrenamiento: foto_mes >= {TRAIN_START_MONTH} Y foto_mes <= {TRAIN_END_MONTH}")
    
    # (Lógica 'all_exclude_months' eliminada)

    # Set de Entrenamiento (¡LÓGICA MODIFICADA!)
    train_idx = (df['foto_mes'] >= TRAIN_START_MONTH) & (df['foto_mes'] <= TRAIN_END_MONTH)
    X_train = df.loc[train_idx, features]
    y_train = df.loc[train_idx, target]

    # Set de Validación (Test)
    val_idx = df['foto_mes'].isin(TEST_MONTHS)
    X_val = df.loc[val_idx, features]
    y_val = df.loc[val_idx, target]

    print(f"Tamaño de Entrenamiento: {X_train.shape[0]} registros")
    print(f"Tamaña de Validación:   {X_val.shape[0]} registros")
    
    if X_val.empty:
        print("\n¡ADVERTENCIA! El set de validación está vacío.")
        sys.exit(1)

    # --- 7. Bucle de Entrenamiento (NUEVO) ---
    
    all_importances = [] # Lista para guardar los DFs de importancia de cada seed

    print(f"\nIniciando entrenamiento en bucle para {len(N_SEEDS)} seeds...")

    for seed in N_SEEDS:
        print(f"\n--- Entrenando con Seed: {seed} ---")
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='none', # <-- MODIFICADO (antes 'auc')
            n_estimators=1000,
            learning_rate=0.05,
            n_jobs=-1,
            seed=seed, # <-- ¡Semilla variable!
            colsample_bytree=0.7,
            subsample=0.7,
            reg_alpha=0.1,
            reg_lambda=0.1,
            is_unbalanced=True
        )

        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=lgbm_profit_metric, # <-- MODIFICADO (antes 'auc')
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature=cat_features_final
        )
        
        # --- MODIFICADO: Reportar Ganancia en lugar de AUC ---
        best_profit = lgb_model.best_score_['valid_0']['max_profit']
        print(f"  Mejor Ganancia (Seed {seed}): ${best_profit:,.0f}")

        # --- 8. Extraer y Guardar Importancias (por Seed) ---
        df_importance_fold = pd.DataFrame({
            'feature': features,
            'importance_gain': lgb_model.feature_importances_
        })
        all_importances.append(df_importance_fold)

    print("\nEntrenamiento de todos los Seeds completado.")

    # --- 9. Agregar Resultados (NUEVO) ---
    print("Agregando importancias de todos los seeds...")
    
    df_all_importances = pd.concat(all_importances)
    
    # Calcular media, std dev, y cuántas veces fue "usada"
    df_agg = df_all_importances.groupby('feature')['importance_gain'].agg(
        mean_importance_gain='mean',
        std_importance_gain='std',
        count_used=lambda x: (x > 0).sum()
    ).reset_index()
    
    df_agg = df_agg.sort_values(by='mean_importance_gain', ascending=False).reset_index(drop=True)
    
    # Guardar en CSV
    df_agg.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)
    
    print(f"\n--- ¡Selección de Features Completada! ---")
    print(f"Lista agregada (promedio, std) guardada en: {IMPORTANCE_OUTPUT_PATH}")
    
    print(f"\n--- TOP 50 FEATURES MÁS INFLUYENTES (Promedio de {len(N_SEEDS)} Seeds) ---")
    print(df_agg.head(50).to_string())
    print("---------------------------------------------------------------")


except Exception as e:
    print(f"\nHa ocurrido un error durante el proceso de selección: {e}")

finally:
    if 'con' in locals() or 'con' in globals():
        con.close()
        print("\nConexión a DuckDB cerrada.")
