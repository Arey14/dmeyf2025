import duckdb
import os
import sys
import pandas as pd

# --- Configuración de Archivos ---
# Asegúrate de que este nombre coincida con tu archivo CSV
CSV_FILE_PATH = 'competencia_02_crudo.csv'
RAW_PARQUET_PATH = CSV_FILE_PATH.rsplit('.', 1)[0] + '.parquet'
REPAIRED_PARQUET_PATH = CSV_FILE_PATH.rsplit('.', 1)[0] + '_reparado.parquet'
# ---------------------------------

print(f"Iniciando el proceso de reparación de datos...")
print(f"Archivo de entrada (CSV): {CSV_FILE_PATH}")
print(f"Archivo de salida (Parquet): {REPAIRED_PARQUET_PATH}")

# Conectar a DuckDB en memoria
con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB (en memoria) establecida.")

try:
    # --- 1. Carga de Datos Crudos ---
    if not os.path.exists(RAW_PARQUET_PATH):
        print(f"Parquet crudo no encontrado. Creando desde '{CSV_FILE_PATH}'...")
        if not os.path.exists(CSV_FILE_PATH):
            raise FileNotFoundError(f"¡El archivo CSV original ({CSV_FILE_PATH}) no se encuentra!")
        
        # Convertir CSV a Parquet para acelerar
        con.execute(f"""
            COPY (SELECT * FROM read_csv_auto('{CSV_FILE_PATH}'))
            TO '{RAW_PARQUET_PATH}' (FORMAT 'PARQUET')
            """)
        print(f"Archivo Parquet crudo '{RAW_PARQUET_PATH}' creado.")
    else:
        print(f"Archivo Parquet crudo encontrado: '{RAW_PARQUET_PATH}'.")

    # Crear la VISTA de datos CRUDOS
    con.execute(f"CREATE VIEW bank_data_raw AS SELECT * FROM read_parquet('{RAW_PARQUET_PATH}')")
    print("Vista 'bank_data_raw' (datos crudos) creada.")

except Exception as e:
    print(f"\nHa ocurrido un error fatal durante la carga de datos: {e}")
    con.close()
    sys.exit(1) # Detener el script si los datos no se pueden cargar


try:
    # --- 2. Creación de la Variable Target (clase_ternaria) ---
    print("Creando la variable target 'clase_ternaria'...")

    query_clases = """
    CREATE OR REPLACE TABLE clases_ternarias AS
    WITH ranked_clients AS (
        SELECT
            foto_mes,
            numero_de_cliente,
            row_number() OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes DESC) as rank
        FROM bank_data_raw
    )
    SELECT
        foto_mes,
        numero_de_cliente,
        CASE
            WHEN rank = 1 AND foto_mes < 202106 THEN 'BAJA+1'
            WHEN rank = 2 AND foto_mes < 202105 THEN 'BAJA+2'
            ELSE 'CONTINUA'
        END AS clase_ternaria
    FROM ranked_clients;
    """

    query_labeled_view = """
    CREATE OR REPLACE VIEW bank_data_labeled AS
    SELECT
        crudo.*,
        clase.clase_ternaria
    FROM bank_data_raw AS crudo
    INNER JOIN clases_ternarias AS clase
        ON crudo.foto_mes = clase.foto_mes AND crudo.numero_de_cliente = clase.numero_de_cliente;
    """
    con.execute(query_clases)
    con.execute(query_labeled_view)
    print("Vista 'bank_data_labeled' (datos + target) creada.")

    
    # --- 3. Script de Reparación y Limpieza (SQL) ---
    print("Iniciando limpieza de datos:")
    print(f" - Excluyendo mes 202006.")
    print(f" - Interpolando gaps en meses problemáticos.")
    print(f" - Corrigiendo anomalías y 'concept drift'.")

    # Lista de columnas a interpolar (aquellas que fueron 0.0 en los meses de gap)
    cols_to_interpolate = [
        'mrentabilidad', 'mrentabilidad_annual', 'mcomisiones', 'mactivos_margen', 
        'mpasivos_margen', 'mcomisiones_otras', 'chomebanking_transacciones', 
        'ctarjeta_visa_debitos_automaticos', 'mttarjeta_visa_debitos_automaticos',
        'ccajeros_propios_descuentos', 'mcajeros_propios_descuentos', 
        'ctarjeta_visa_descuentos', 'mtarjeta_visa_descuentos', 
        'ctarjeta_master_descuentos', 'mtarjeta_master_descuentos'
        # Se agregan más meses de gaps de descuentos identificados en el EDA
    ]

    # Meses donde se detectaron los gaps (excluyendo 202006 que se elimina)
    meses_gap = [201905, 201910, 201904, 202002, 202009, 202010, 202102] 

    # 1. Crear CTE que convierte los 0.0 anómalos en NULL
    sql_cte_nullif = "WITH base_filtered AS (\n    SELECT \n        *,\n"
    for col in cols_to_interpolate:
        # Usamos TRY_CAST por si la columna fue leída como VARCHAR
        sql_cte_nullif += f"        NULLIF(TRY_CAST({col} AS DOUBLE), 0) AS {col}_null,\n"
    sql_cte_nullif = sql_cte_nullif.rstrip(',\n') + "\n"
    sql_cte_nullif += "    FROM bank_data_labeled\n    WHERE foto_mes != 202006\n),\n" # <-- Exclusión de 202006

    # 2. Crear CTE que obtiene el valor anterior (LAG) y siguiente (LEAD)
    sql_cte_lag_lead = "imputation_data AS (\n    SELECT \n        *,\n"
    for col in cols_to_interpolate:
        # --- MODIFICADO --- Se quitó IGNORE NULLS
        sql_cte_lag_lead += f"        LAG({col}_null) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS prev_{col},\n"
        sql_cte_lag_lead += f"        LEAD({col}_null) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS next_{col},\n"
    sql_cte_lag_lead = sql_cte_lag_lead.rstrip(',\n') + "\n    FROM base_filtered\n)\n" # <-- MODIFICADO: Añadido un \n al final

    # 3. Construir la consulta final (SELECT)
    sql_final_select = "SELECT \n" # <-- MODIFICADO: Quitado "CREATE OR REPLACE TABLE..."
    
    # Obtener todas las columnas de la tabla original para el SELECT final
    all_cols_df = con.execute("DESCRIBE bank_data_labeled").fetchdf()
    original_cols = all_cols_df['column_name'].tolist()

    # --- MANEJO DE CONCEPT DRIFT ---
    # 1. Quitar cmobile_app_trx, la reemplazaremos por dos columnas nuevas
    if 'cmobile_app_trx' in original_cols:
        original_cols.remove('cmobile_app_trx')
    
    # 2. Añadir las nuevas columnas al final de la consulta
    new_drift_cols = """
    -- Corregir tmobile_app (datos nulos antes de 201907)
    CASE 
        WHEN foto_mes < 201907 THEN NULL 
        ELSE tmobile_app 
    END AS tmobile_app,
    
    -- Dividir cmobile_app_trx en dos variables debido al cambio de definición
    -- Variable 1: Conteo de transacciones (hasta 2020-09)
    CASE 
        WHEN foto_mes < 202010 THEN cmobile_app_trx 
        ELSE NULL 
    END AS cmobile_app_trx_pre202010,
    
    -- Variable 2: Flag binario de uso (desde 2020-10)
    CASE 
        WHEN foto_mes >= 202010 THEN cmobile_app_trx 
        ELSE NULL 
    END AS cmobile_app_trx_post202010,
    """
    # -------------------------------

    for col in original_cols:
        if col in cols_to_interpolate:
            # Esta columna necesita ser reparada (interpolación)
            sql_final_select += f"""
            CASE 
                WHEN foto_mes IN ({', '.join(map(str, meses_gap))}) AND {col}_null IS NULL
                THEN (COALESCE(prev_{col}, next_{col}, 0) + COALESCE(next_{col}, prev_{col}, 0)) / 2
                ELSE {col}
            END AS {col},
            """
        
        # --- MANEJO DE ANOMALÍAS ---
        elif col == 'ccaja_seguridad':
            sql_final_select += "    CASE WHEN ccaja_seguridad > 1 THEN 1 ELSE ccaja_seguridad END AS ccaja_seguridad,\n"
        
        elif col == 'cliente_edad':
            sql_final_select += "    CASE WHEN cliente_edad > 100 OR cliente_edad < 18 THEN NULL ELSE cliente_edad END AS cliente_edad,\n"
        
        elif 'Fvencimiento' in col: # Captura Master_Fvencimiento y Visa_Fvencimiento
            sql_final_select += f"    CASE WHEN {col} < -20000 THEN NULL ELSE {col} END AS {col},\n"
        # ---------------------------

        # --- MANEJO DE INCONSISTENCIAS LÓGICAS ---
        elif col == 'cprestamos_prendarios':
            sql_final_select += "    CASE WHEN cprestamos_prendarios = 0 AND mprestamos_prendarios > 0 THEN 1 ELSE cprestamos_prendarios END AS cprestamos_prendarios,\n"
        
        elif col == 'cpayroll_trx':
            sql_final_select += "    CASE WHEN cpayroll_trx = 0 AND mpayroll > 0 THEN 1 ELSE cpayroll_trx END AS cpayroll_trx,\n"
        # -------------------------------------
        
        elif col == 'tmobile_app': # Esta la manejamos en new_drift_cols
            pass
        
        else:
            # Columna normal, solo seleccionarla
            sql_final_select += f"    {col},\n"
            
    sql_final_select += new_drift_cols # Añadir las nuevas columnas de drift
    sql_final_select = sql_final_select.rstrip(',\n') + "\nFROM imputation_data;"

    # Combinar todas las partes de la consulta SQL
    # --- MODIFICADO: "CREATE TABLE..." se mueve al inicio de la query ---
    full_repair_query = "CREATE OR REPLACE TABLE data_reparada AS\n" + sql_cte_nullif + sql_cte_lag_lead + sql_final_select
    
    # print("\n--- Ejecutando consulta de reparación ---")
    # print(full_repair_query) # Descomentar para depurar la consulta
    # print("--------------------------------------")

    con.execute(full_repair_query)
    print("Tabla 'data_reparada' creada exitosamente.")

    # --- 4. Guardar los Datos Reparados ---
    print(f"Guardando datos reparados en '{REPAIRED_PARQUET_PATH}'...")
    con.execute(f"COPY (SELECT * FROM data_reparada) TO '{REPAIRED_PARQUET_PATH}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
    
    print("\n--- ¡Proceso de reparación completado! ---")
    print(f"Archivo final: {REPAIRED_PARQUET_PATH}")
    
    # Verificar el esquema final
    print("\nEsquema del archivo reparado:")
    con.execute(f"DESCRIBE SELECT * FROM '{REPAIRED_PARQUET_PATH}' LIMIT 1")
    print(con.fetchdf())


except Exception as e:
    print(f"\nHa ocurrido un error durante el proceso de reparación: {e}")

finally:
    con.close()
    print("Conexión a DuckDB cerrada.")
