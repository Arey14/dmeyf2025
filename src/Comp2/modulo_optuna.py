import duckdb
import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import json

# --- 1. Definición de Ganancia ---
PROFIT_TP = 780000  # Ganancia por Verdadero Positivo (detectar un BAJA+2)
COST_FP = -20000    # Costo por Falso Positivo (molestar a un cliente CONTINUA)
# ---------------------------------

# --- 2. Métrica de Ganancia Personalizada (para LGBM) ---
def lgbm_profit_metric(y_true, y_pred_probs):
    """
    Métrica de ganancia personalizada para LightGBM.
    Encuentra el umbral que maximiza la ganancia.
    """
    y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    max_profit = -np.inf
    best_thresh = 0.5
    thresholds = np.linspace(0.01, 0.50, 50)
    
    for thresh in thresholds:
        y_pred = (y_pred_probs > thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_profit = (tp * PROFIT_TP) + (fp * COST_FP)
        
        if total_profit > max_profit:
            max_profit = total_profit
            best_thresh = thresh
            
    # Guardamos el mejor umbral en un atributo estático
    lgbm_profit_metric.best_thresh = best_thresh
    return 'max_profit', max_profit, True
# ---------------------------------

# --- 3. Configuración de Archivos y Parámetros ---
FEATURES_PARQUET_PATH = 'dataset_202004_202108.parquet'
IMPORTANCE_INPUT_PATH = 'feature_importances_conjunto.json'
TOP_N_FEATURES = 1330 # Usar todas las features
N_TRIALS_OPTUNA = 100  # 50 trials por cada estudio (total 100)
N_ESTIMATORS = 1000    # n_estimators
GLOBAL_EXCLUDE_MONTHS = [ 202006 ] # Mes anómalo

# --- CAMBIO: Lista de variables a excluir ---
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
# -----------------------------------------------

# --- CAMBIO: Definimos las semillas a usar para la robustez en cada trial ---
SEEDS_FOR_ROBUSTNESS = [761249, 762001, 763447, 762233, 761807,800003, 800021, 800051, 800093, 800123] # CAMBIO: Usar 10 semillas.
UNDERSAMPLING_RATIO = 15 # Ratio fijo de continuas vs bajas
# --------------------------------------------------------------------------
# --- CAMBIO: Base de datos de Optuna para persistencia ---
OPTUNA_DB_PATH = "sqlite:///optuna_studies.db"
# --- FIN DEL CAMBIO ---
# -----------------------------------------

print(f"--- Iniciando Optimización de Hiperparámetros con Optuna ---")
print(f"Dataset de entrada: {FEATURES_PARQUET_PATH}")
print(f"Features de entrada: {IMPORTANCE_INPUT_PATH} (Top {TOP_N_FEATURES})")

# --- 5. Preparación de Datos (Funciones Helper) ---
def fix_dtypes_and_target(df):
    """Aplica dtypes, casting y mapeo de target."""
    if 'cmobile_app_trx_pre202010' in df.columns:
        df['cmobile_app_trx_pre202010'] = pd.to_numeric(df['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df.columns:
        df['cmobile_app_trx_post202010'] = pd.to_numeric(df['cmobile_app_trx_post202010'], errors='coerce').fillna(0)
    
    target = 'clase_ternaria'
    df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
    df[target] = df[target].fillna(0).astype(int)
    return df, target

def get_features_and_cats(df, top_features_list):
    """Obtiene la lista de features y columnas categóricas."""
    # Usar solo las Top N features
    features = [f for f in top_features_list if f in df.columns]
    
    cat_features = [
        'anio', 'mes', 'peor_estado_tarjetas', 'active_quarter', 'cliente_vip',
        'internet', 'tcallcenter', 'thomebanking', 'tmobile_app', 
        'Master_delinquency', 'Visa_delinquency', 'ccaja_seguridad'
    ]
    
    cat_features_final = []
    for col in cat_features:
        if col in df.columns and col in features: # Solo si es una feature Top N
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].astype('category')
            cat_features_final.append(col)
            
    return features, cat_features_final
# -------------------------------------------------

# --- 6. Definición de la Función Objetivo de Optuna ---

# Variables globales para que la función 'objective' las vea
# Se re-definirán en cada llamada a 'run_optimization_study'
X_train, y_train, X_val, y_val, cat_features_final_global = [None] * 5
bajas_global, continuas_global = [None] * 2 # --- CAMBIO: Añadido para optimización de memoria

# --- CAMBIO: La función objective ahora es robusta ---
def objective(trial):
    """
    Función que Optuna intentará maximizar.
    Prueba un set de hiperparámetros y devuelve la ganancia PROMEDIO
    de varias semillas para asegurar robustez.
    """
    global X_train, y_train, X_val, y_val, cat_features_final_global
    global bajas_global, continuas_global # --- CAMBIO: Añadido para optimización de memoria
    
    # 1. Definir el Espacio de Búsqueda de Hiperparámetros
    params = {
        'objective': 'binary',
        'metric': 'none',
        'n_estimators': N_ESTIMATORS, 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_bin': 31,
        'n_jobs': -1,
        # 'seed': 42, # <-- SE QUITA LA SEMILLA FIJA DE AQUÍ
        # 'is_unbalanced': True, # <-- CAMBIO: Se quita, ahora hacemos undersampling manual
        
        # Parámetros a optimizar
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
    }

    # 2. Bucle de entrenamiento por semilla para robustez
    profits_per_seed = []
    
    for seed in SEEDS_FOR_ROBUSTNESS:
        # --- CAMBIO: Lógica de Undersampling Optimizada (Usa globales) ---
        # Ya no se hace: train_data = pd.concat([X_train, y_train], axis=1)
        # Ya no se hace: bajas = train_data[train_data[target_col_name] == 1]
        # Ya no se hace: continuas = train_data[train_data[target_col_name] == 0]
        
        # Aseguramos no pedir más muestras de las que hay
        n_continuas_to_keep = min(int(len(bajas_global) * UNDERSAMPLING_RATIO), len(continuas_global))
        
        continuas_undersampled = continuas_global.sample(n=n_continuas_to_keep, random_state=seed)
        
        # Combinar y barajar el dataset de entrenamiento submuestreado
        train_undersampled = pd.concat([bajas_global, continuas_undersampled]).sample(frac=1, random_state=seed)
        
        X_train_us = train_undersampled[X_train.columns]
        y_train_us = train_undersampled[y_train.name] # Usar el nombre de la columna target
        # --- FIN CAMBIO ---

        # Asignar la semilla de esta iteración
        params['seed'] = seed
        
        lgb_model = lgb.LGBMClassifier(**params)
        
        lgb_model.fit(
            X_train_us, y_train_us, # <-- CAMBIO: Usar datos con undersampling
            eval_set=[(X_val, y_val)],
            eval_metric=lgbm_profit_metric, # ¡Optimizamos por ganancia!
            callbacks=[lgb.early_stopping(100, verbose=False)], # (paciencia de 100)
            categorical_feature=cat_features_final_global
        )
        
        # Guardar la mejor ganancia para esta semilla
        best_profit_for_this_seed = lgb_model.best_score_['valid_0']['max_profit']
        profits_per_seed.append(best_profit_for_this_seed)
    
    # 3. Devolver la ganancia PROMEDIO
    # Optuna ahora maximizará la ganancia media de las 3 semillas
    return np.mean(profits_per_seed)
# --- FIN DEL CAMBIO ---

# -------------------------------------------------

def run_optimization_study(con, top_features_list, train_start, train_end, val_month, output_json_name):
    """
    Ejecuta un estudio completo de Optuna para una ventana de tiempo específica.
    """
    global X_train, y_train, X_val, y_val, cat_features_final_global
    global bajas_global, continuas_global # --- CAMBIO: Añadido para optimización de memoria

    print(f"\n==========================================================")
    print(f"--- Iniciando Estudio para '{output_json_name}' ---")
    print(f"  Entrenamiento: {train_start} - {train_end} (excl. {GLOBAL_EXCLUDE_MONTHS})")
    print(f"  Validación:     {val_month}")
    print(f"==========================================================")

    # --- Carga y Preparación de Datos ---
    months_to_load = [val_month] + list(range(train_start, train_end + 1))
    
    sql_load_data = f"""
    SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')
    WHERE foto_mes IN ({', '.join(map(str, months_to_load))})
    AND foto_mes NOT IN ({', '.join(map(str, GLOBAL_EXCLUDE_MONTHS))})
    """
    
    print(f"Cargando datos para el estudio (Meses {train_start} a {val_month})...")
    df = con.execute(sql_load_data).fetchdf()
    
    # Preparar el DataFrame (Dtypes, Target, etc.)
    df, target = fix_dtypes_and_target(df)
    features, cat_features_final_global = get_features_and_cats(df, top_features_list)
    
    # --- Crear Splits (UNA SOLA VEZ) ---
    print("Creando sets de Entrenamiento y Validación...")
    
    train_idx = (df['foto_mes'] >= train_start) & (df['foto_mes'] <= train_end)
    X_train = df.loc[train_idx, features]
    y_train = df.loc[train_idx, target]

    val_idx = df['foto_mes'] == val_month
    X_val = df.loc[val_idx, features]
    y_val = df.loc[val_idx, target]
    
    # --- CAMBIO: Pre-calcular bajas y continuas para optimizar memoria ---
    print("Pre-calculando DataFrames de undersampling (bajas/continuas)...")
    target_col_name = y_train.name
    # Usamos X_train y y_train que ya están en memoria
    train_data = pd.concat([X_train, y_train], axis=1) 
    bajas_global = train_data[train_data[target_col_name] == 1].copy()
    continuas_global = train_data[train_data[target_col_name] == 0].copy()
    del train_data # Liberar esta copia intermedia
    print(f"  - Bajas: {len(bajas_global)} | Continuas: {len(continuas_global)}")
    # --- FIN CAMBIO ---

    print(f"  Tamaño de Entrenamiento: {X_train.shape[0]} registros")
    print(f"  Tamaño de Validación:    {X_val.shape[0]} registros")
    
    del df # Liberar memoria del dataframe completo
    
    # --- Iniciar el Estudio Optuna ---
    # --- CAMBIO: Mensaje actualizado para reflejar el cambio ---
    print(f"\nIniciando estudio Optuna ({N_TRIALS_OPTUNA} trials)...")
    print(f"Optimizando para maximizar la 'max_profit' PROMEDIO (sobre {len(SEEDS_FOR_ROBUSTNESS)} semillas por trial).")
    # --- FIN DEL CAMBIO ---
    
    # Desactivar logs de Optuna para que no sature la consola
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # --- CAMBIO: Configuración de persistencia de Optuna ---
    # Usar un nombre de estudio único basado en el archivo de salida
    study_name_for_db = f"study_{output_json_name.replace('.json', '')}"
    
    print(f"Usando base de datos de persistencia: {OPTUNA_DB_PATH}")
    print(f"Nombre del estudio en la DB: {study_name_for_db}")
    
    study = optuna.create_study(
        study_name=study_name_for_db,
        storage=OPTUNA_DB_PATH,
        direction='maximize',
        load_if_exists=True  # <-- Esto permite retomar el estudio
    )
    # --- FIN DEL CAMBIO ---
    
    # --- CAMBIO: Mostrar cuántos trials faltan ---
    # Comprobar si el estudio ya tiene trials de ejecuciones anteriores
    n_trials_ya_hechos = len(study.trials)
    n_trials_a_ejecutar = N_TRIALS_OPTUNA - n_trials_ya_hechos

    if n_trials_ya_hechos > 0:
        print(f"Estudio cargado. Ya se han completado {n_trials_ya_hechos} trials.")
        
    if n_trials_a_ejecutar > 0:
        print(f"Se ejecutarán {n_trials_a_ejecutar} nuevos trials...")
        study.optimize(objective, n_trials=n_trials_a_ejecutar, show_progress_bar=True)
    else:
        print(f"El estudio ya ha completado los {N_TRIALS_OPTUNA} trials. No se ejecutarán nuevos trials.")
    # --- FIN DEL CAMBIO ---

    print("\n--- ¡Optimización Completada! ---")
    print(f"\nMejor Ganancia (max_profit) PROMEDIO encontrada: ${study.best_value:,.0f}")
    
    print("\nMejores Hiperparámetros encontrados:")
    print(study.best_params)
    
    # Guardar los mejores parámetros en un JSON
    with open(output_json_name, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nMejores parámetros guardados en '{output_json_name}'")


def main():
    
    if not os.path.exists(FEATURES_PARQUET_PATH) or not os.path.exists(IMPORTANCE_INPUT_PATH):
        print(f"Error: Faltan archivos. Asegúrate de que '{FEATURES_PARQUET_PATH}' y '{IMPORTANCE_INPUT_PATH}' existan.")
        sys.exit(1)

    con = duckdb.connect(database=':memory:', read_only=False)
    print("Conexión a DuckDB establecida.")
    
    try:
        # Cargar la lista de Top N features
        df_top_features = pd.read_json(IMPORTANCE_INPUT_PATH) # Usar JSON
        top_features_list_raw = df_top_features.head(TOP_N_FEATURES)['feature'].tolist()

        # --- CAMBIO: Filtrar la lista de features ---
        top_features_list = [f for f in top_features_list_raw if f not in VARIABLES_TO_EXCLUDE]
        
        num_excluded = len(top_features_list_raw) - len(top_features_list)
        print(f"Cargadas {len(top_features_list_raw)} features top. Excluyendo {num_excluded} variables.")
        print(f"Número final de features a usar en el estudio: {len(top_features_list)}")
        # --- FIN DEL CAMBIO ---

        # --- ESTUDIO 1: Para validar en ABRIL 2021 (Gap 1 mes) ---
        run_optimization_study(
            con=con,
            top_features_list=top_features_list, # Se pasa la lista ya filtrada
            train_start=202004,
            train_end=202102,
            val_month=202104,
            output_json_name="best_hyperparams_val_202104.json"
        )
        
        # --- ESTUDIO 2: Para validar en JUNIO 2021 (Gap 1 mes) ---
        run_optimization_study(
            con=con,
            top_features_list=top_features_list, # Se pasa la lista ya filtrada
            train_start=202007,
            train_end=202104,
            val_month=202106,
            output_json_name="best_hyperparams_val_202106.json"
        )

    except Exception as e:
        print(f"\nHa ocurrido un error durante el proceso de optimización: {e}")

    finally:
        if 'con' in locals() or 'con' in globals():
            con.close()
            print("\nConexión a DuckDB cerrada.")


if __name__ == "__main__":
    main()
