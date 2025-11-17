import duckdb
import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
import json
import gc

# --- 1. Definición de Ganancia (SOLO PARA INFO, NO SE USA PARA CÁLCULO) ---
PROFIT_TP = 780000
COST_FP = -20000
# ---------------------------------

# --- 2. Métrica de Ganancia Personalizada (para LGBM) ---
# Esta métrica es necesaria para el early_stopping
def lgbm_profit_metric(y_true, y_pred_probs):
    y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    max_profit = -np.inf
    thresholds = np.linspace(0.01, 0.50, 50)
    
    for thresh in thresholds:
        y_pred = (y_pred_probs > thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_profit = (tp * PROFIT_TP) + (fp * COST_FP)
        
        if total_profit > max_profit:
            max_profit = total_profit
            
    return 'max_profit', max_profit, True
# ---------------------------------

# --- 3. Configuración Principal del Script ---
# --- Archivo de hiperparámetros a cargar ---
HYPERPARAMS_JSON_PATH = 'best_hyperparams_junio.json'

# --- Meses de Entrenamiento y Predicción ---
TRAIN_START_MONTH = 202005
TRAIN_END_MONTH = 202105 # El entrenamiento real
VALIDATION_MONTH = 202106 # Se usa para el early_stopping
PREDICTION_MONTH = 202108

# --- Cutoff fijo para la sumisión (basado en apo-506.pdf, pág 7) ---
N_CLIENTES_A_ENVIAR = 13000

# --- Configuración de Datos (traída de modulo_optuna_robusto.py) ---
# --- CAMBIO: Ahora apunta al archivo pre-procesado ---
FEATURES_PARQUET_PATH = 'preprocessed_final_data.parquet'
# --- FIN DEL CAMBIO ---
IMPORTANCE_INPUT_PATH = 'feature_importances_conjunto.json'
TOP_N_FEATURES = 1330
GLOBAL_EXCLUDE_MONTHS = [ 202006 ] # Mes anómalo

# --- Lista de variables a excluir ---
VARIABLES_TO_EXCLUDE = [
    'rank_pct_cf_mprestamos_personales', 'mprestamos_personales', 'ratio_drift_mprestamos_personales_1m',
    'ratio_mprestamos_personales_vs_avg6m', 'ratio_drift_mprestamos_personales_3m', 'avg_mprestamos_personales_1m',
    'avg_mprestamos_personales_3m', 'delta_mprestamos_personales_6m', 'ratio_drift_mprestamos_personales_6m',
    'delta_mprestamos_personales_3m', 'rank_pct_cf_cprestamos_personales', 'avg_cprestamos_personales_6m',
    'max_mprestamos_personales_6m', 'delta_mprestamos_personales_1m', 'max_cprestamos_personales_6m',
    'sum_cprestamos_personales_6m', 'avg_mprestamos_personales_6m', 'avg_cprestamos_personales_3m',
    'cprestamos_personales', 'ratio_cprestamos_personales_vs_avg6m', 'ratio_drift_cprestamos_personales_3m',
    'delta_cprestamos_personales_3m', 'sum_mprestamos_personales_6m', 'ratio_drift_cprestamos_personales_6m',
    'ratio_drift_cprestamos_personales_1m', 'delta_cprestamos_personales_6m', 'avg_cprestamos_personales_1m',
    'delta_cprestamos_personales_1m'
]

# --- Semillas para el "Semillerio" final ---
SEEDS_FOR_FINAL_MODEL = [761249, 762001, 763447, 762233, 761807,800003, 800021, 800051, 800093, 800123] # 10 semillas.

# --- Estrategia de datos (Robusto / LGBM Normal) ---
UNDERSAMPLING_RATIO = 150 # Ratio de 'modulo_optuna_robusto.py'
TARGET_INCLUYE_BAJA1 = False # Target de 'modulo_optuna_robusto.py'
N_ESTIMATORS = 1000 # Max de 'modulo_optuna_robusto.py'
# -----------------------------------------

print(f"--- Iniciando Script de Predicción Final (Estrategia Robusta / LGBM Normal) ---")
print(f"  Entrenando en: {TRAIN_START_MONTH} - {TRAIN_END_MONTH}")
print(f"  Validando en:  {VALIDATION_MONTH} (para early stopping)")
print(f"  Prediciendo en: {PREDICTION_MONTH}")
print(f"  Hiperparámetros cargados de: {HYPERPARAMS_JSON_PATH}")

# --- 4. Preparación de Datos (Funciones Helper) ---
def fix_dtypes_and_target(df):
    """Aplica dtypes, casting y mapeo de target."""
    if 'cmobile_app_trx_pre202010' in df.columns:
        df['cmobile_app_trx_pre202010'] = pd.to_numeric(df['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df.columns:
        df['cmobile_app_trx_post202010'] = pd.to_numeric(df['cmobile_app_trx_post202010'], errors='coerce').fillna(0)
    
    target = 'clase_ternaria'
    # Target de la estrategia "Robusta"
    if TARGET_INCLUYE_BAJA1:
        df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 1, 'CONTINUA': 0})
    else:
        df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
    # Los meses de predicción (ej. 202108) tendrán NaN, que .fillna(0) convierte a 'CONTINUA'
    df[target] = df[target].fillna(0).astype(int) 
    return df, target

def get_features_and_cats(df, top_features_list):
    """Obtiene la lista de features y columnas categóricas."""
    features = [f for f in top_features_list if f in df.columns]
    
    cat_features = [
        'anio', 'mes', 'peor_estado_tarjetas', 'active_quarter', 'cliente_vip',
        'internet', 'tcallcenter', 'thomebanking', 'tmobile_app', 
        'Master_delinquency', 'Visa_delinquency', 'ccaja_seguridad'
    ]
    
    cat_features_final = []
    for col in cat_features:
        if col in df.columns and col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].astype('category')
            cat_features_final.append(col)
            
    return features, cat_features_final
# -------------------------------------------------

def main():
    
    # --- Validaciones iniciales ---
    if not os.path.exists(HYPERPARAMS_JSON_PATH):
        print(f"Error: No se encuentra el archivo de hiperparámetros '{HYPERPARAMS_JSON_PATH}'.")
        print("Asegúrate de que el JSON del script 'modulo_optuna_robusto.py' exista.")
        sys.exit(1)
        
    # --- CAMBIO: No es necesario comprobar IMPORTANCE_INPUT_PATH si el parquet ya está procesado ---
    # ---         Pero lo mantenemos porque el script AÚN LO USA para definir la lista de features ---
    if not os.path.exists(FEATURES_PARQUET_PATH) or not os.path.exists(IMPORTANCE_INPUT_PATH):
        print(f"Error: Faltan archivos de datos ('{FEATURES_PARQUET_PATH}' o '{IMPORTANCE_INPUT_PATH}').")
        print(f"Asegúrate de ejecutar primero 'create_preprocessed_parquet.py'")
        sys.exit(1)

    con = duckdb.connect(database=':memory:', read_only=False)
    print("\nConexión a DuckDB establecida.")
    
    try:
        # --- 1. Cargar Hiperparámetros y Features ---
        print(f"Cargando hiperparámetros de '{HYPERPARAMS_JSON_PATH}'...")
        with open(HYPERPARAMS_JSON_PATH, 'r') as f:
            best_params = json.load(f)
        
        print("Hiperparámetros cargados:")
        print(best_params)
        
        df_top_features = pd.read_json(IMPORTANCE_INPUT_PATH)
        top_features_list_raw = df_top_features.head(TOP_N_FEATURES)['feature'].tolist()
        top_features_list = [f for f in top_features_list_raw if f not in VARIABLES_TO_EXCLUDE]
        print(f"Número final de features a usar: {len(top_features_list)}")

        # --- 2. Carga y Preparación de Datos ---
        # --- CAMBIO: La consulta ahora es mucho más simple ---
        # Ya no filtramos por meses aquí, el parquet ya viene filtrado
        sql_load_data = f"""
        SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')
        """
        
        print(f"\nCargando datos (Entrenamiento, Validación y Predicción)...")
        df = con.execute(sql_load_data).fetchdf()
        
        df, target = fix_dtypes_and_target(df)
        features, cat_features_final = get_features_and_cats(df, top_features_list)
        
        # --- 3. Crear Splits ---
        print("Creando sets de Entrenamiento, Validación y Predicción...")
        
        train_idx = (df['foto_mes'] >= TRAIN_START_MONTH) & (df['foto_mes'] <= TRAIN_END_MONTH)
        df_train = df.loc[train_idx].copy()

        valid_idx = (df['foto_mes'] == VALIDATION_MONTH)
        df_valid = df.loc[valid_idx].copy()
        
        future_idx = (df['foto_mes'] == PREDICTION_MONTH)
        df_future = df.loc[future_idx].copy()
        
        del df # Liberar memoria del dataframe completo
        
        # Preparar datos de predicción (X_future)
        X_future = df_future[features]
        future_ids = df_future[['numero_de_cliente', 'foto_mes']]
        del df_future
        
        # Preparar datos de validación (X_val, y_val)
        X_val = df_valid[features]
        y_val = df_valid[target]
        del df_valid
        
        # Preparar datos de entrenamiento (para undersampling)
        print("Pre-calculando DataFrames de undersampling (bajas/continuas)...")
        bajas_global = df_train[df_train[target] == 1].copy()
        continuas_global = df_train[df_train[target] == 0].copy()
        del df_train
        print(f"  - Positivos (Target Robusto): {len(bajas_global)} | Negativos: {len(continuas_global)}")

        gc.collect()

        # --- 4. Loop de Entrenamiento Final ("Semillerio") ---
        print(f"\n--- Iniciando Entrenamiento del Semillerio ({len(SEEDS_FOR_FINAL_MODEL)} modelos) ---")
        
        predictions_accumulator = np.zeros(len(X_future))
        
        for i, seed in enumerate(SEEDS_FOR_FINAL_MODEL):
            print(f"  Entrenando modelo {i+1}/{len(SEEDS_FOR_FINAL_MODEL)} (Seed: {seed})...")
            
            # --- Parámetros (cargados de Optuna) ---
            params = best_params.copy() # Cargar los hiperparámetros optimizados
            
            # --- Añadir parámetros fijos/de control ---
            params['n_estimators'] = N_ESTIMATORS
            params['metric'] = 'none' # Usamos el eval_metric personalizado
            params['n_jobs'] = -1
            params['seed'] = seed # Semilla de esta iteración
            
            # --- Undersampling para esta semilla ---
            n_continuas_to_keep = min(int(len(bajas_global) * UNDERSAMPLING_RATIO), len(continuas_global))
            continuas_undersampled = continuas_global.sample(n=n_continuas_to_keep, random_state=seed)
            train_undersampled = pd.concat([bajas_global, continuas_undersampled]).sample(frac=1, random_state=seed)
            
            X_train_us = train_undersampled[features]
            y_train_us = train_undersampled[target]
            
            # --- Entrenar el modelo ---
            lgb_model = lgb.LGBMClassifier(**params)
            
            lgb_model.fit(
                X_train_us, 
                y_train_us,
                eval_set=[(X_val, y_val)], # Usamos el set de validación
                eval_metric=lgbm_profit_metric,
                callbacks=[lgb.early_stopping(100, verbose=False)],
                categorical_feature=cat_features_final
            )
            
            # --- Predecir y Acumular ---
            val_probs = lgb_model.predict_proba(X_future)[:, 1]
            predictions_accumulator += val_probs
            
            # Liberar memoria del loop
            del lgb_model, X_train_us, y_train_us, train_undersampled, continuas_undersampled
            gc.collect()

        # --- 5. Generar Predicción Final ---
        print("\n--- Entrenamiento completado. Generando predicción final ---")
        
        # Promediar las predicciones del semillerio
        final_predictions = predictions_accumulator / len(SEEDS_FOR_FINAL_MODEL)
        
        df_submission = future_ids.copy()
        df_submission['prob'] = final_predictions
        
        # Ordenar por probabilidad descendente
        df_submission = df_submission.sort_values(by='prob', ascending=False)
        
        # Crear la columna 'Predicted'
        df_submission['Predicted'] = 0
        
        # Marcar los N_CLIENTES_A_ENVIAR como 1
        df_submission.iloc[:N_CLIENTES_A_ENVIAR, df_submission.columns.get_loc('Predicted')] = 1
        
        # Seleccionar columnas finales
        output_df = df_submission[['numero_de_cliente', 'Predicted']]
        
        # Guardar el archivo
        output_filename = f"submission_{PREDICTION_MONTH}.csv"
        output_df.to_csv(output_filename, index=False)
        
        print(f"\n¡Éxito! Archivo '{output_filename}' generado.")
        print(f"  - Total de clientes en predicción: {len(output_df)}")
        print(f"  - Clientes marcados para enviar (Predicted=1): {output_df['Predicted'].sum()} (de {N_CLIENTES_A_ENVIAR})")

    except Exception as e:
        print(f"\nHa ocurrido un error durante el proceso de predicción: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'con' in locals() or 'con' in globals():
            con.close()
            print("\nConexión a DuckDB cerrada.")


if __name__ == "__main__":
    main()
