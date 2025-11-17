import duckdb
import os
import sys
import pandas as pd

# --- 1. CONFIGURACIÓN MODULAR ---
# ¡IMPORTANTE! Esta configuración debe ser IDÉNTICA
# a la que se usó en 'feature_engineering.py' para que
# el script sepa qué columnas debe verificar.
FE_CONFIG = {
    "crear_combinaciones_basicas": True,
    "crear_deltas_y_tendencias": True,
    "crear_features_categoricas": True,
    "crear_ratios_avanzados": True,
    "crear_rankings_percentiles": True,
}

# Estas listas también deben coincidir con las del script de FE
TIME_WINDOWS = [1, 3, 6]
COLS_TO_PROCESS_DELTAS = [
    'mcuentas_saldo', 'mrentabilidad', 'ctrx_quarter', 
    'mtarjeta_visa_consumo', 'mtarjeta_master_consumo', 'cproductos', 
    'ctotal_debitos_automaticos', 'ctotal_cheques_rechazados'
]
COLS_TO_TRACK_SIMPLE = [
    'active_quarter', 'cliente_vip', 'internet', 'tcallcenter', 
    'thomebanking', 'tmobile_app', 'Master_delinquency', 'Visa_delinquency', 
    'ccaja_seguridad', 'ctotal_debitos_automaticos', 'ctotal_cheques_rechazados'
]
COLS_TO_RATIO_DRIFT = [
    'mcuentas_saldo', 'mrentabilidad', 'ctrx_quarter', 'mconsumo_total_tarjetas'
]
COLS_TO_RANK = [
    'mrentabilidad', 'mcuentas_saldo', 'mconsumo_total_tarjetas', 
    'ctrx_quarter', 'cproductos', 'ctotal_debitos_automaticos'
]
STATUS_COLS_TO_TRACK = ['Master_status', 'Visa_status', 'peor_estado_tarjetas']
# ---------------------------------


# --- Configuración de Archivos ---
FEATURES_PARQUET_PATH = 'features_dataset.parquet'
# ---------------------------------

print(f"--- Iniciando Verificación de Feature Engineering ---")
print(f"Cargando archivo: {FEATURES_PARQUET_PATH}\n")

if not os.path.exists(FEATURES_PARQUET_PATH):
    print(f"Error: No se encontró el archivo de features '{FEATURES_PARQUET_PATH}'.")
    print("Asegúrate de que 'feature_engineering.py' se haya ejecutado exitosamente.")
    sys.exit(1)

# Conectar a DuckDB en memoria
con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB establecida.")

# Lista para guardar todos los tests
tests_pasados = 0
tests_fallados = 0

try:
    # Cargar los datos reparados en una vista para consultar
    con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')")
    print("Vista 'data' (dataset de features) creada.\n")
    print("--- Ejecutando Pruebas de Features ---\n")
    
    all_cols_df = con.execute("DESCRIBE data").fetchdf()
    all_columns = all_cols_df['column_name'].tolist()

    # --- 1. Verificación de Combinaciones Básicas ---
    print("1. Verificando Módulo [Combinaciones Básicas]:")
    if FE_CONFIG["crear_combinaciones_basicas"]:
        expected_cols = ['anio', 'mes', 'mconsumo_total_tarjetas', 'peor_estado_tarjetas', 'ctotal_debitos_automaticos']
        missing_cols = [col for col in expected_cols if col not in all_columns]
        if not missing_cols:
            print(f"   [OK] Columnas de combinaciones básicas (ej. 'peor_estado_tarjetas') existen.\n")
            tests_pasados += 1
        else:
            print(f"   [FALLO] Faltan columnas de combinaciones básicas: {missing_cols}\n")
            tests_fallados += 1
    else:
        print("   [INFO] Módulo 'crear_combinaciones_basicas' fue omitido (según config).\n")


    # --- 2. Verificación de Deltas y Tendencias ---
    print("2. Verificando Módulo [Deltas y Tendencias]:")
    if FE_CONFIG["crear_deltas_y_tendencias"]:
        # Muestra de columnas esperadas
        expected_cols = [
            f'delta_{COLS_TO_PROCESS_DELTAS[0]}_{TIME_WINDOWS[0]}m', # ej. delta_mcuentas_saldo_1m
            f'avg_{COLS_TO_PROCESS_DELTAS[1]}_{TIME_WINDOWS[1]}m',   # ej. avg_mrentabilidad_3m
            'stddev_mcuentas_saldo_3m',
            'count_meses_mora_master_6m'
        ]
        missing_cols = [col for col in expected_cols if col not in all_columns]
        if not missing_cols:
            print(f"   [OK] Columnas de deltas y tendencias (ej. 'delta_...', 'avg_...', 'stddev_...') existen.\n")
            tests_pasados += 1
        else:
            print(f"   [FALLO] Faltan columnas de deltas y tendencias: {missing_cols}\n")
            tests_fallados += 1
    else:
        print("   [INFO] Módulo 'crear_deltas_y_tendencias' fue omitido (según config).\n")

    # --- 3. Verificación de Features Categóricas ---
    print("3. Verificando Módulo [Features Categóricas]:")
    if FE_CONFIG["crear_features_categoricas"]:
        expected_cols = [
            f'cambio_{COLS_TO_TRACK_SIMPLE[0]}', # ej. cambio_active_quarter
            f'dejo_{COLS_TO_TRACK_SIMPLE[-1]}', # ej. dejo_ctotal_cheques_rechazados
            f'inicio_cierre_{STATUS_COLS_TO_TRACK[0]}' # ej. inicio_cierre_Master_status
        ]
        missing_cols = [col for col in expected_cols if col not in all_columns]
        if not missing_cols:
            print(f"   [OK] Columnas de eventos categóricos (ej. 'cambio_...', 'dejo_...', 'inicio_cierre_...') existen.\n")
            tests_pasados += 1
        else:
            print(f"   [FALLO] Faltan columnas de eventos categóricos: {missing_cols}\n")
            tests_fallados += 1
    else:
        print("   [INFO] Módulo 'crear_features_categoricas' fue omitido (según config).\n")


    # --- 4. Verificación de Ratios Avanzados ---
    print("4. Verificando Módulo [Ratios Avanzados]:")
    if FE_CONFIG["crear_ratios_avanzados"]:
        expected_cols = [
            'ratio_comisiones_rentabilidad',
            f'ratio_drift_{COLS_TO_RATIO_DRIFT[0]}_{TIME_WINDOWS[0]}m' # ej. ratio_drift_mcuentas_saldo_1m
        ]
        missing_cols = [col for col in expected_cols if col not in all_columns]
        if not missing_cols:
            print(f"   [OK] Columnas de ratios (ej. 'ratio_comisiones_rentabilidad', 'ratio_drift_...') existen.\n")
            tests_pasados += 1
        else:
            print(f"   [FALLO] Faltan columnas de ratios: {missing_cols}\n")
            tests_fallados += 1
    else:
        print("   [INFO] Módulo 'crear_ratios_avanzados' fue omitido (según config).\n")

    # --- 5. Verificación de Rankings Percentiles ---
    print("5. Verificando Módulo [Rankings Percentiles]:")
    if FE_CONFIG["crear_rankings_percentiles"]:
        col_to_check = f'rank_pct_{COLS_TO_RANK[0]}' # ej. rank_pct_mrentabilidad
        
        if col_to_check not in all_columns:
            print(f"   [FALLO] Columnas de ranking (ej. '{col_to_check}') NO existen.\n")
            tests_fallados += 1
        else:
            print(f"   [OK] Columnas de ranking (ej. '{col_to_check}') existen.")
            
            # Verificar el rango de valores
            min_rank, max_rank = con.execute(f"SELECT MIN({col_to_check}), MAX({col_to_check}) FROM data").fetchone()
            
            if min_rank >= 0 and max_rank <= 1:
                print(f"   [OK] Valores de ranking están en el rango correcto [0, 1] (Min: {min_rank:.2f}, Max: {max_rank:.2f}).\n")
                tests_pasados += 1
            else:
                print(f"   [FALLO] Valores de ranking están FUERA del rango [0, 1] (Min: {min_rank}, Max: {max_rank}).\n")
                tests_fallados += 1
    else:
        print("   [INFO] Módulo 'crear_rankings_percentiles' fue omitido (según config).\n")

    # --- 6. Verificación de Imputación de NULLs ---
    print("6. Verificando Imputación de NULLs (¡Crítico!):")
    # Verificamos que las features generadas por ventanas (que crean NULLs)
    # hayan sido rellenadas por la función 'get_final_select_list'.
    
    imputacion_fallida = 0
    
    # Test 1: Deltas (imputados a 0)
    if FE_CONFIG["crear_deltas_y_tendencias"]:
        col_delta = f'delta_{COLS_TO_PROCESS_DELTAS[0]}_{TIME_WINDOWS[0]}m' # ej. delta_mcuentas_saldo_1m
        null_count = con.execute(f"SELECT COUNT(*) FROM data WHERE {col_delta} IS NULL").fetchone()[0]
        if null_count == 0:
            print(f"   [OK] Features 'delta_...' (ej. {col_delta}) no contienen NULLs.")
            tests_pasados += 1
        else:
            print(f"   [FALLO] ¡{col_delta} contiene {null_count} NULLs! La imputación a 0 falló.")
            imputacion_fallida += 1

    # Test 2: Ratios de Drift (imputados a 1)
    if FE_CONFIG["crear_ratios_avanzados"]:
        col_ratio = f'ratio_drift_{COLS_TO_RATIO_DRIFT[0]}_{TIME_WINDOWS[0]}m' # ej. ratio_drift_mcuentas_saldo_1m
        null_count = con.execute(f"SELECT COUNT(*) FROM data WHERE {col_ratio} IS NULL").fetchone()[0]
        if null_count == 0:
            print(f"   [OK] Features 'ratio_drift_...' (ej. {col_ratio}) no contienen NULLs.")
            tests_pasados += 1
        else:
            print(f"   [FALLO] ¡{col_ratio} contiene {null_count} NULLs! La imputación a 1 falló.")
            imputacion_fallida += 1

    # Test 3: Rankings (imputados a 0.5)
    if FE_CONFIG["crear_rankings_percentiles"]:
        col_rank = f'rank_pct_{COLS_TO_RANK[0]}' # ej. rank_pct_mrentabilidad
        null_count = con.execute(f"SELECT COUNT(*) FROM data WHERE {col_rank} IS NULL").fetchone()[0]
        if null_count == 0:
            print(f"   [OK] Features 'rank_pct_...' (ej. {col_rank}) no contienen NULLs.")
            tests_pasados += 1
        else:
            print(f"   [FALLO] ¡{col_rank} contiene {null_count} NULLs! La imputación a 0.5 falló.")
            imputacion_fallida += 1

    if imputacion_fallida > 0:
        tests_fallados += imputacion_fallida
    
    if imputacion_fallida == 0 and (FE_CONFIG["crear_deltas_y_tendencias"] or FE_CONFIG["crear_ratios_avanzados"] or FE_CONFIG["crear_rankings_percentiles"]):
        print(f"   [INFO] Verificación de imputación de NULLs completada.\n")
    elif not FE_CONFIG["crear_deltas_y_tendencias"] and not FE_CONFIG["crear_ratios_avanzados"] and not FE_CONFIG["crear_rankings_percentiles"]:
        print("   [INFO] No se generaron features que requieran imputación (deltas, ratios, rankings).\n")
    else:
        print("\n")


except Exception as e:
    print(f"\nHa ocurrido un error durante el proceso de verificación: {e}")
    tests_fallados += 1

finally:
    con.close()
    print("Conexión a DuckDB cerrada.")
    
    print("\n--- Resumen de la Verificación ---")
    print(f"Total Pruebas Pasadas: {tests_pasados}")
    print(f"Total Pruebas Falladas: {tests_fallados}")
    if tests_fallados == 0:
        print("\n[ÉXITO] El dataset de features pasó todas las verificaciones.")
    else:
        print("\n[ERROR] El dataset de features falló una o más verificaciones.")
    print("-----------------------------------")
