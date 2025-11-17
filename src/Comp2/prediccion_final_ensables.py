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
# HYPERPARAMS_JSON_PATH = 'best_hyperparams_junio.json' # <-- ELIMINAMOS ESTA CONSTANTE GLOBAL

# --- Mes de Predicción (Constante) ---
PREDICTION_MONTH = 202108

# --- NUEVO: Meses de Entrenamiento y Validación fijos ---
TRAINING_MONTHS = [
    202004, 202005, 202007, 202008, 202009, 202010,
    202011, 202012, 202101, 202102, 202103, 202104
]
VALIDATION_MONTH = 202106 # Validación constante

# --- NUEVO: Configuración de los Runs del Ensamble ---
# Ahora solo definimos los hiperparámetros, ya que los meses son fijos.
ENSEMBLE_CONFIGS = [
    {
        'id': 'run_all_months_junio',
        'hyperparams_file': 'best_hyperparams_junio.json'
    },
    {
        'id': 'run_all_months_abril',
        'hyperparams_file': 'best_hyperparams_abril.json'
    },
]
# --- FIN DE LA NUEVA CONFIGURACIÓN ---


# --- Cutoff fijo para la sumisión (basado en apo-506.pdf, pág 7) ---
N_CLIENTES_A_ENVIAR = 13000

# --- Configuración de Datos (traída de modulo_optuna_robusto.py) ---
FEATURES_PARQUET_PATH = 'preprocessed_final_data.parquet'
IMPORTANCE_INPUT_PATH = 'feature_importances_conjunto.json'
TOP_N_FEATURES = 1330
GLOBAL_EXCLUDE_MONTHS = [ 202006 ] # Mes anómalo (se usa en pre-procesamiento, no aquí)

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

# --- Semillas para el "Semillerio" (se usa en CADA run del ensamble) ---
SEEDS_FOR_FINAL_MODEL = [761249, 762001, 763447, 762233, 761807, 800003, 800021, 800051, 800093, 800123] # 10 semillas.

# --- Estrategia de datos (Robusto / LGBM Normal) ---
UNDERSAMPLING_RATIO = 150
TARGET_INCLUYE_BAJA1 = False
N_ESTIMATORS = 1000
# -----------------------------------------

print(f"--- Iniciando Script de Predicción (Estrategia de Ensamble por Meses) ---")
print(f"Prediciendo en: {PREDICTION_MONTH}")
# print(f"Hiperparámetros cargados de: {HYPERPARAMS_JSON_PATH}") # <-- ELIMINADO
print(f"Número de Runs en el Ensamble: {len(ENSEMBLE_CONFIGS)}")
print(f"Número de Semillas por Run: {len(SEEDS_FOR_FINAL_MODEL)}")
print(f"Total de modelos a entrenar: {len(ENSEMBLE_CONFIGS) * len(SEEDS_FOR_FINAL_MODEL)}")

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


# --- NUEVA FUNCIÓN ---
def train_semillerio_run(
    bajas_run: pd.DataFrame, 
    continuas_run: pd.DataFrame, 
    X_val_run: pd.DataFrame, 
    y_val_run: pd.Series, 
    X_future_run: pd.DataFrame,
    features: list, 
    cat_features_final: list, 
    target: str, 
    best_params: dict
    ):
    """
    Ejecuta un "semillerio" completo para un conjunto de datos (bajas/continuas)
    y un set de validación (X_val/y_val) dados.
    
    Predice sobre X_future_run.
    Retorna las predicciones promediadas del semillerio.
    """
    
    print(f"    Iniciando semillerio... (Positivos: {len(bajas_run)}, Negativos: {len(continuas_run)})")
    
    predictions_accumulator = np.zeros(len(X_future_run))
    
    for i, seed in enumerate(SEEDS_FOR_FINAL_MODEL):
        print(f"      Entrenando modelo {i+1}/{len(SEEDS_FOR_FINAL_MODEL)} (Seed: {seed})...")
        
        # --- Parámetros (cargados de Optuna) ---
        params = best_params.copy() # Cargar los hiperparámetros optimizados
        
        # --- Añadir parámetros fijos/de control ---
        params['n_estimators'] = N_ESTIMATORS
        params['metric'] = 'none' # Usamos el eval_metric personalizado
        params['n_jobs'] = -1
        params['seed'] = seed # Semilla de esta iteración
        
        # --- Undersampling para esta semilla ---
        n_continuas_to_keep = min(int(len(bajas_run) * UNDERSAMPLING_RATIO), len(continuas_run))
        continuas_undersampled = continuas_run.sample(n=n_continuas_to_keep, random_state=seed)
        train_undersampled = pd.concat([bajas_run, continuas_undersampled]).sample(frac=1, random_state=seed)
        
        X_train_us = train_undersampled[features]
        y_train_us = train_undersampled[target]
        
        # --- Entrenar el modelo ---
        lgb_model = lgb.LGBMClassifier(**params)
        
        lgb_model.fit(
            X_train_us, 
            y_train_us,
            eval_set=[(X_val_run, y_val_run)], # Usamos el set de validación
            eval_metric=lgbm_profit_metric,
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature=cat_features_final
        )
        
        # --- Predecir y Acumular ---
        val_probs = lgb_model.predict_proba(X_future_run)[:, 1]
        predictions_accumulator += val_probs
        
        # Liberar memoria del loop
        del lgb_model, X_train_us, y_train_us, train_undersampled, continuas_undersampled
        gc.collect()

    # Promediar las predicciones del semillerio
    final_predictions_for_run = predictions_accumulator / len(SEEDS_FOR_FINAL_MODEL)
    print(f"    Semillerio completado.")
    return final_predictions_for_run
# --- FIN NUEVA FUNCIÓN ---


# --- FUNCIÓN PRINCIPAL (MODIFICADA) ---
def main():
    
    # --- Validaciones iniciales ---
    # Ya no comprobamos HYPERPARAMS_JSON_PATH aquí
    # ...
    # if not os.path.exists(HYPERPARAMS_JSON_PATH):
    #     print(f"Error: No se encuentra el archivo de hiperparámetros '{HYPERPARAMS_JSON_PATH}'.")
    #     sys.exit(1)
        
    if not os.path.exists(FEATURES_PARQUET_PATH) or not os.path.exists(IMPORTANCE_INPUT_PATH):
        print(f"Error: Faltan archivos de datos ('{FEATURES_PARQUET_PATH}' o '{IMPORTANCE_INPUT_PATH}').")
        sys.exit(1)

    con = duckdb.connect(database=':memory:', read_only=False)
    print("\nConexión a DuckDB establecida.")
    
    try:
        # --- 1. Cargar Hiperparámetros y Features ---
        
        # --- MODIFICADO: No cargamos un solo JSON, pre-cargamos TODOS ---
        print("\nPre-cargando todos los hiperparámetros del ensamble...")
        
        # 1. Encontrar todos los archivos de hiperparámetros únicos
        hyperparams_files_to_load = set(config['hyperparams_file'] for config in ENSEMBLE_CONFIGS)
        hyperparams_cache = {}
        
        for param_file in hyperparams_files_to_load:
            if not os.path.exists(param_file):
                print(f"Error: No se encuentra el archivo de hiperparámetros '{param_file}' definido en ENSEMBLE_CONFIGS.")
                sys.exit(1)
            
            with open(param_file, 'r') as f:
                hyperparams_cache[param_file] = json.load(f)
            print(f"  - '{param_file}' cargado en caché.")
        # --- FIN DE MODIFICACIÓN ---
        
        df_top_features = pd.read_json(IMPORTANCE_INPUT_PATH)
        top_features_list_raw = df_top_features.head(TOP_N_FEATURES)['feature'].tolist()
        top_features_list = [f for f in top_features_list_raw if f not in VARIABLES_TO_EXCLUDE]
        print(f"Número final de features a usar: {len(top_features_list)}")

        # --- 2. Carga y Preparación de Datos (Completo) ---
        # Cargamos TODOS los datos relevantes a la memoria una sola vez
        sql_load_data = f"""
        SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')
        """
        
        print(f"\nCargando datos (Completos)...")
        df = con.execute(sql_load_data).fetchdf()
        
        df, target = fix_dtypes_and_target(df)
        features, cat_features_final = get_features_and_cats(df, top_features_list)
        
        # --- 3. Crear Split de PREDICCIÓN (constante para todos los runs) ---
        print(f"Creando set de Predicción (para {PREDICTION_MONTH})...")
        future_idx = (df['foto_mes'] == PREDICTION_MONTH)
        df_future = df.loc[future_idx].copy()
        
        X_future = df_future[features]
        future_ids = df_future[['numero_de_cliente', 'foto_mes']]
        del df_future
        gc.collect()

        # --- NUEVO: 3.5. Crear Splits de TRAIN y VALID (constantes para todos los runs) ---
        print(f"\nCreando set de Validación (para {VALIDATION_MONTH})...")
        valid_idx = (df['foto_mes'] == VALIDATION_MONTH)
        df_valid = df.loc[valid_idx].copy()
        
        if df_valid.empty:
            print(f"Error: No hay datos de validación para el mes {VALIDATION_MONTH}. Saliendo.")
            sys.exit(1)
            
        X_val_run = df_valid[features]
        y_val_run = df_valid[target]
        del df_valid
        
        print(f"Creando set de Entrenamiento (con {len(TRAINING_MONTHS)} meses)...")
        # Usamos la lista TRAINING_MONTHS directamente
        train_idx = df['foto_mes'].isin(TRAINING_MONTHS)
        df_train = df.loc[train_idx].copy()
        
        if df_train.empty:
            print("Error: No hay datos de entrenamiento para los meses especificados. Saliendo.")
            sys.exit(1)

        # Pre-preparar los dataframes de undersampling (se usarán en ambos runs)
        print("Pre-calculando DataFrames de undersampling (bajas/continuas)...")
        bajas_run_global = df_train[df_train[target] == 1].copy()
        continuas_run_global = df_train[df_train[target] == 0].copy()
        del df_train
        gc.collect()
        print(f"  - Positivos (Target Robusto): {len(bajas_run_global)} | Negativos: {len(continuas_run_global)}")
        # --- FIN DE LA NUEVA LÓGICA DE SPLITS ---
        
        
        # --- 4. Loop de Ensamble Principal ---
        print(f"\n--- Iniciando Loop de Ensamble ({len(ENSEMBLE_CONFIGS)} runs) ---")
        
        global_predictions_list = [] # Acumulador para los promedios de CADA run
        
        for config in ENSEMBLE_CONFIGS:
            print(f"\n  Iniciando Run: '{config['id']}'")
            # print(f"  Config: Train={config['train_start']}-{config['train_end']}, Valid={config['validation']}") # <-- Ya no aplica
            print(f"  Usando {len(TRAINING_MONTHS)} meses de train y validación en {VALIDATION_MONTH}")
            
            # --- NUEVO: Obtener los hiperparámetros para ESTE run ---
            param_file = config['hyperparams_file']
            run_best_params = hyperparams_cache[param_file]
            print(f"  Usando hiperparámetros de: '{param_file}'")
            # --- FIN NUEVO ---

            # --- LÓGICA DE SPLITS ELIMINADA DE AQUÍ ---
            # Ya no creamos splits por cada run, usamos los globales
            
            # --- Llamar a la función del semillerio ---
            # Esta función entrenará N semillas y devolverá 1 array de predicciones (promediadas)
            run_predictions = train_semillerio_run(
                bajas_run=bajas_run_global,           # <-- Usamos el DF global
                continuas_run=continuas_run_global, # <-- Usamos el DF global
                X_val_run=X_val_run,                  # <-- Usamos el X_val global
                y_val_run=y_val_run,                  # <-- Usamos el y_val global
                X_future_run=X_future, # Siempre predecimos sobre el mismo X_future
                features=features,
                cat_features_final=cat_features_final,
                target=target,
                best_params=run_best_params # <-- MODIFICADO: Pasar los params del run
            )
            
            # Acumular el resultado de este run
            global_predictions_list.append(run_predictions)
            
            # Ya no borramos los DFs de train/val, se reúsan
            # del bajas_run, continuas_run, X_val_run, y_val_run, run_predictions
            del run_predictions # Solo borramos la predicción de este loop
            gc.collect()

        # --- 5. Generar Predicción Final del Ensamble ---
        if not global_predictions_list:
            print("Error: No se completó ningún run del ensamble. No se puede generar la sumisión.")
            sys.exit(1)
            
        print("\n--- Todos los runs completados. Generando predicción final del ensamble ---")
        
        # Promediar las predicciones de TODOS los runs
        final_predictions = np.mean(np.array(global_predictions_list), axis=0)
        
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
        output_filename = f"submission_ensemble_{PREDICTION_MONTH}.csv"
        output_df.to_csv(output_filename, index=False)
        
        print(f"\n¡Éxito! Archivo '{output_filename}' generado.")
        print(f"  - Total de clientes en predicción: {len(output_df)}")
        print(f"  - Clientes marcados para enviar (Predicted=1): {output_df['Predicted'].sum()} (de {N_CLIENTES_A_ENVIAR})")
        print(f"  - Promedio de probabilidad (Top {N_CLIENTES_A_ENVIAR}): {df_submission.head(N_CLIENTES_A_ENVIAR)['prob'].mean():.6f}")
        print(f"  - Mínima probabilidad (Top {N_CLIENTES_A_ENVIAR}): {df_submission.head(N_CLIENTES_A_ENVIAR)['prob'].min():.6f}")


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
