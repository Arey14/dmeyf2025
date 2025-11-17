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

# --- 2. Métrica de Ganancia (Cálculo Manual) ---
# Se usa para cálculo manual post-entrenamiento
def calculate_max_profit(y_true, y_pred_probs):
    """
    Calcula la ganancia máxima encontrando el mejor umbral.
    """
    max_profit = -np.inf
    thresholds = np.linspace(0.01, 0.50, 50)
    
    for thresh in thresholds:
        y_pred = (y_pred_probs > thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_profit = (tp * PROFIT_TP) + (fp * COST_FP)
        
        if total_profit > max_profit:
            max_profit = total_profit
            
    return max_profit
# ---------------------------------

# --- 3. Configuración de Archivos y Parámetros ---
FEATURES_PARQUET_PATH = 'dataset_202004_202108.parquet'
IMPORTANCE_INPUT_PATH = 'feature_importances_conjunto.json'
TOP_N_FEATURES = 1330
N_TRIALS_OPTUNA = 500
N_ESTIMATORS = 9999 # Fijo del PDF (num_iterations)
GLOBAL_EXCLUDE_MONTHS = [ 202006 ]

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

# --- Semillas para robustez ---
SEEDS_FOR_ROBUSTNESS = [761249] #, 762001, 763447, 762233, 761807,800003, 800021, 800051, 800093, 800123] # 10 semillas.

# --- Estrategia de datos del PDF ---
UNDERSAMPLING_FRACTION = 0.30 # 30% de las filas CONTINUA (del PDF)
TARGET_INCLUYE_BAJA1 = True # (del PDF)

# --- Base de datos de Optuna para este experimento ---
OPTUNA_DB_PATH = "sqlite:///optuna_studies_zcustom.db"
# -----------------------------------------

print(f"--- Iniciando Optimización (Estrategia z-Custom) ---")
print(f"Dataset: {FEATURES_PARQUET_PATH}")
print(f"Features: {IMPORTANCE_INPUT_PATH} (Top {TOP_N_FEATURES})")

# --- 5. Preparación de Datos (Funciones Helper) ---
def fix_dtypes_and_target(df):
    """Aplica dtypes, casting y mapeo de target."""
    if 'cmobile_app_trx_pre202010' in df.columns:
        df['cmobile_app_trx_pre202010'] = pd.to_numeric(df['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df.columns:
        df['cmobile_app_trx_post202010'] = pd.to_numeric(df['cmobile_app_trx_post202010'], errors='coerce').fillna(0)
    
    target = 'clase_ternaria'
    # Target adaptado al PDF (BAJA+1 y BAJA+2 son positivos)
    if TARGET_INCLUYE_BAJA1:
        df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 1, 'CONTINUA': 0})
    else:
        df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
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

# --- 6. Definición de la Función Objetivo de Optuna ---

X_train, y_train, X_val, y_val, cat_features_final_global = [None] * 5
bajas_global, continuas_global, X_val_global_copy = [None] * 3

def objective(trial):
    """
    Función que Optuna intentará maximizar.
    Prueba un set de hiperparámetros (canaritos, gradient_bound, feature_fraction)
    y devuelve la ganancia PROMEDIO de varias semillas.
    NO USA EARLY STOPPING.
    """
    global X_train, y_train, X_val, y_val, cat_features_final_global
    global bajas_global, continuas_global, X_val_global_copy
    
    # 1. Definir el Espacio de Búsqueda de Hiperparámetros (según tu solicitud)
    qcanaritos = trial.suggest_int('canaritos', 50, 100) # 0 a 10 canaritos
    gradient_bound = trial.suggest_float('gradient_bound', 0.1, 0.5) # Rango de prueba
    feature_fraction = trial.suggest_float('feature_fraction', 0.4, 0.99) # Rango de prueba
    
    # 2. Parámetros Fijos (del PDF)
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'none', # 'custom' es R, 'none' es Python
        'first_metric_only': False,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_row_wise': True,
        'verbosity': -1, # -100 es R, -1 es Python
        
        'max_bin': 31,
        'min_data_in_leaf': 20,
        
        'num_iterations': N_ESTIMATORS, # 9999
        'num_leaves': 999,
        'learning_rate': 1.0,
        
        'n_jobs': -1,
        
        # --- Parámetros de tu fork custom (que se están optimizando) ---
        'canaritos': qcanaritos,
        'gradient_bound': gradient_bound,
        'feature_fraction': feature_fraction
    }

    # 3. Bucle de entrenamiento por semilla para robustez
    profits_per_seed = []
    
    # Hacemos una copia de X_val para añadir canaritos (no modificar la global)
    X_val_canaries = X_val_global_copy.copy()
    
    # --- Añadir canaritos a X_val (se hace una vez por trial) ---
    canary_cols_val = {}
    for i in range(qcanaritos):
        col_name = f'canarito_{i}'
        canary_cols_val[col_name] = np.random.rand(len(X_val_canaries))
    
    if qcanaritos > 0:
        X_val_canaries = pd.concat([
            pd.DataFrame(canary_cols_val, index=X_val_canaries.index), 
            X_val_canaries
        ], axis=1)

    
    for seed in SEEDS_FOR_ROBUSTNESS:
        
        # Lógica de Undersampling (del PDF)
        n_continuas_to_keep = int(len(continuas_global) * UNDERSAMPLING_FRACTION)
        continuas_undersampled = continuas_global.sample(n=n_continuas_to_keep, random_state=seed)
        
        train_undersampled = pd.concat([bajas_global, continuas_undersampled]).sample(frac=1, random_state=seed)
        
        X_train_us = train_undersampled[X_train.columns]
        y_train_us = train_undersampled[y_train.name]
        
        # --- Añadir canaritos a X_train_us (se hace en cada semilla) ---
        X_train_us_canaries = X_train_us.copy()
        canary_cols_train = {}
        for i in range(qcanaritos):
            col_name = f'canarito_{i}'
            canary_cols_train[col_name] = np.random.rand(len(X_train_us_canaries))
        
        if qcanaritos > 0:
             X_train_us_canaries = pd.concat([
                pd.DataFrame(canary_cols_train, index=X_train_us_canaries.index), 
                X_train_us_canaries
            ], axis=1)

        # Asignar la semilla de esta iteración
        params['seed'] = seed
        
        lgb_model = lgb.LGBMClassifier(**params)
        
        # --- CAMBIO: Fit SIN early stopping ---
        # Las columnas categóricas se mantienen, los canaritos son numéricos
        lgb_model.fit(
            X_train_us_canaries, 
            y_train_us,
            categorical_feature=cat_features_final_global
        )
        
        # --- CAMBIO: Cálculo de ganancia manual ---
        # Usamos el X_val con canaritos que preparamos fuera del bucle
        val_probs = lgb_model.predict_proba(X_val_canaries)[:, 1]
        
        # Usamos y_val (el target original)
        profit = calculate_max_profit(y_val, val_probs)
        profits_per_seed.append(profit)
    
    # 4. Devolver la ganancia PROMEDIO
    return np.mean(profits_per_seed)
# -------------------------------------------------

def run_optimization_study(con, top_features_list, train_start, train_end, val_month, output_json_name):
    """
    Ejecuta un estudio completo de Optuna para una ventana de tiempo específica.
    """
    global X_train, y_train, X_val, y_val, cat_features_final_global
    global bajas_global, continuas_global, X_val_global_copy

    print(f"\n==========================================================")
    print(f"--- Iniciando Estudio para '{output_json_name}' ---")
    print(f"  Entrenamiento: {train_start} - {train_end} (excl. {GLOBAL_EXCLUDE_MONTHS})")
    print(f"  Validación:     {val_month}")
    print(f"  ESTRATEGIA: z-Custom (lr=1.0, no-early-stop, canaritos, gradient_bound)")
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
    
    # Hacemos una copia de X_val para usar en 'objective'
    X_val_global_copy = X_val.copy()
    
    print("Pre-calculando DataFrames de undersampling (bajas/continuas)...")
    target_col_name = y_train.name
    train_data = pd.concat([X_train, y_train], axis=1) 
    bajas_global = train_data[train_data[target_col_name] == 1].copy()
    continuas_global = train_data[train_data[target_col_name] == 0].copy()
    del train_data
    
    print(f"  - Positivos (Target PDF): {len(bajas_global)} | Negativos: {len(continuas_global)}")
    print(f"  Tamaño de Entrenamiento: {X_train.shape[0]} registros")
    print(f"  Tamamño de Validación:    {X_val.shape[0]} registros")
    
    del df
    
    # --- Iniciar el Estudio Optuna ---
    print(f"\nIniciando estudio Optuna ({N_TRIALS_OPTUNA} trials)...")
    print(f"Optimizando para maximizar la 'max_profit' PROMEDIO (sobre {len(SEEDS_FOR_ROBUSTNESS)} semillas por trial).")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study_name_for_db = f"study_zcustom_{output_json_name.replace('.json', '')}"
    
    print(f"Usando base de datos de persistencia: {OPTUNA_DB_PATH}")
    print(f"Nombre del estudio en la DB: {study_name_for_db}")
    
    study = optuna.create_study(
        study_name=study_name_for_db,
        storage=OPTUNA_DB_PATH,
        direction='maximize',
        load_if_exists=True
    )
    
    n_trials_ya_hechos = len(study.trials)
    n_trials_a_ejecutar = N_TRIALS_OPTUNA - n_trials_ya_hechos

    if n_trials_ya_hechos > 0:
        print(f"Estudio cargado. Ya se han completado {n_trials_ya_hechos} trials.")
        
    if n_trials_a_ejecutar > 0:
        print(f"Se ejecutarán {n_trials_a_ejecutar} nuevos trials...")
        study.optimize(objective, n_trials=n_trials_a_ejecutar, show_progress_bar=True)
    else:
        print(f"El estudio ya ha completado los {N_TRIALS_OPTUNA} trials. No se ejecutarán nuevos trials.")

    print("\n--- ¡Optimización Completada! ---")
    print(f"\nMejor Ganancia (max_profit) PROMEDIO encontrada: ${study.best_value:,.0f}")
    
    print("\nMejores Hiperparámetros encontrados:")
    print(study.best_params)
    
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
        df_top_features = pd.read_json(IMPORTANCE_INPUT_PATH)
        top_features_list_raw = df_top_features.head(TOP_N_FEATURES)['feature'].tolist()

        top_features_list = [f for f in top_features_list_raw if f not in VARIABLES_TO_EXCLUDE]
        
        num_excluded = len(top_features_list_raw) - len(top_features_list)
        print(f"Cargadas {len(top_features_list_raw)} features top. Excluyendo {num_excluded} variables.")
        print(f"Número final de features a usar en el estudio: {len(top_features_list)}")

        # --- ESTUDIO 1: Valida en 202104 ---
        run_optimization_study(
            con=con,
            top_features_list=top_features_list,
            train_start=202004,
            train_end=202102,
            val_month=202104,
            output_json_name="best_hyperparams_zcustom_val_202104.json"
        )
        
        # --- ESTUDIO 2: Valida en 202106 ---
        run_optimization_study(
            con=con,
            top_features_list=top_features_list,
            train_start=202004,
            train_end=202104,
            val_month=202106,
            output_json_name="best_hyperparams_zcustom_val_202106.json"
        )

    except Exception as e:
        print(f"\nHa ocurrido un error durante el proceso de optimización: {e}")
        # Imprimir más detalles del error si es posible
        import traceback
        traceback.print_exc()

    finally:
        if 'con' in locals() or 'con' in globals():
            con.close()
            print("\nConexión a DuckDB cerrada.")


if __name__ == "__main__":
    main()
