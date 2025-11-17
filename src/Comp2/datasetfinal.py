import duckdb
import os
import sys
import pandas as pd
import json

# --- 1. Configuración de Archivos de Entrada y Salida ---
FEATURES_PARQUET_PATH_IN = 'features_dataset.parquet'
IMPORTANCE_INPUT_PATH = 'feature_importances_conjunto.json'
FEATURES_PARQUET_PATH_OUT = 'preprocessed_final_data.parquet'

# --- 2. Definición de Meses y Features a Incluir ---
# (Copiados de final_prediction_robusto.py)

# --- Meses de Entrenamiento y Predicción ---
TRAIN_START_MONTH = 202005
TRAIN_END_MONTH = 202105
VALIDATION_MONTH = 202106
PREDICTION_MONTH = 202108
GLOBAL_EXCLUDE_MONTHS = [ 202006 ]

# --- Configuración de Features ---
TOP_N_FEATURES = 1330
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

# Columnas "Core" que no son features pero son necesarias para el script
CORE_COLS_TO_KEEP = ['numero_de_cliente', 'foto_mes', 'clase_ternaria']

# Columnas "Helper" que las funciones de dtypes necesitan
HELPER_COLS_TO_KEEP = ['cmobile_app_trx_pre202010', 'cmobile_app_trx_post202010']

# --- 3. Script Principal de Pre-procesamiento ---
def create_preprocessed_file():
    print("--- Iniciando Script de Pre-procesamiento ---")
    
    # --- Validaciones iniciales ---
    if not os.path.exists(FEATURES_PARQUET_PATH_IN) or not os.path.exists(IMPORTANCE_INPUT_PATH):
        print(f"Error: Faltan archivos de entrada ('{FEATURES_PARQUET_PATH_IN}' o '{IMPORTANCE_INPUT_PATH}').")
        sys.exit(1)
        
    con = duckdb.connect(database=':memory:', read_only=False)
    print("Conexión a DuckDB establecida.")
    
    try:
        # --- 1. Cargar y filtrar lista de features ---
        print(f"Cargando lista de features de '{IMPORTANCE_INPUT_PATH}'...")
        df_top_features = pd.read_json(IMPORTANCE_INPUT_PATH)
        top_features_list_raw = df_top_features.head(TOP_N_FEATURES)['feature'].tolist()
        top_features_list = [f for f in top_features_list_raw if f not in VARIABLES_TO_EXCLUDE]
        print(f"Se usarán {len(top_features_list)} features.")

        # --- 2. Definir meses y columnas finales ---
        train_months = list(range(TRAIN_START_MONTH, TRAIN_END_MONTH + 1))
        months_to_load_list = [PREDICTION_MONTH, VALIDATION_MONTH] + train_months
        
        # Filtrar meses excluidos
        final_months_list = [m for m in months_to_load_list if m not in GLOBAL_EXCLUDE_MONTHS]
        final_months_str = ', '.join(map(str, final_months_list))
        
        # Combinar todas las columnas necesarias
        all_cols_to_keep = set(CORE_COLS_TO_KEEP + HELPER_COLS_TO_KEEP + top_features_list)
        
        # DuckDB no se lleva bien con los nombres de columnas que empiezan con números,
        # así que los escapamos con comillas dobles.
        final_cols_str = ', '.join([f'"{col}"' for col in all_cols_to_keep])

        print(f"Meses a incluir: {final_months_list}")
        print(f"Total de columnas a incluir: {len(all_cols_to_keep)}")

        # --- 3. Construir y ejecutar la consulta SQL ---
        sql_query = f"""
        COPY (
            SELECT {final_cols_str}
            FROM read_parquet('{FEATURES_PARQUET_PATH_IN}')
            WHERE foto_mes IN ({final_months_str})
        ) TO '{FEATURES_PARQUET_PATH_OUT}' (FORMAT PARQUET, COMPRESSION 'ZSTD');
        """
        
        print(f"\nEjecutando consulta para crear '{FEATURES_PARQUET_PATH_OUT}'...")
        con.execute(sql_query)
        
        print("\n¡Éxito! Archivo pre-procesado ha sido creado.")

    except Exception as e:
        print(f"\nHa ocurrido un error durante el pre-procesamiento: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'con' in locals() or 'con' in globals():
            con.close()
            print("\nConexión a DuckDB cerrada.")

if __name__ == "__main__":
    create_preprocessed_file()
