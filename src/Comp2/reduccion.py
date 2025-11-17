import duckdb
import os
import sys

# --- 1. Configuración de Archivos ---
# El dataset completo que generó 'feature_engineering.py'
INPUT_FEATURES_PATH = 'features_dataset.parquet' 

# El nombre del nuevo archivo que se creará
OUTPUT_SUBSET_PATH = 'dataset_202004_202108.parquet'

# --- 2. Configuración de Fechas ---
START_MONTH = 202004
END_MONTH = 202108
# ---------------------------------

print(f"--- Iniciando Creación de Subset de Datos ---")
print(f"Dataset de entrada: {INPUT_FEATURES_PATH}")
print(f"Dataset de salida:  {OUTPUT_SUBSET_PATH}")
print(f"Rango de fechas:    {START_MONTH} a {END_MONTH}\n")

if not os.path.exists(INPUT_FEATURES_PATH):
    print(f"Error: No se encontró el archivo de features '{INPUT_FEATURES_PATH}'.")
    print("Asegúrate de que 'feature_engineering.py' se haya ejecutado exitosamente.")
    sys.exit(1)

# Conectar a DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB establecida.")

try:
    # Usamos COPY para hacer la operación de lectura, filtrado y escritura
    # en un solo paso, lo cual es extremadamente eficiente.
    sql_query = f"""
    COPY (
        SELECT * FROM read_parquet('{INPUT_FEATURES_PATH}')
        WHERE foto_mes >= {START_MONTH} AND foto_mes <= {END_MONTH}
    ) 
    TO '{OUTPUT_SUBSET_PATH}' 
    (FORMAT 'PARQUET', CODEC 'ZSTD');
    """
    
    print("Ejecutando consulta para filtrar y guardar el subset...")
    con.execute(sql_query)
    
    print("\n--- ¡Subset creado exitosamente! ---")
    
    # Verificación
    count_original = con.execute(f"SELECT COUNT(*) FROM read_parquet('{INPUT_FEATURES_PATH}')").fetchone()[0]
    count_subset = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUTPUT_SUBSET_PATH}')").fetchone()[0]
    
    print(f"Registros en dataset original: {count_original}")
    print(f"Registros en nuevo subset:   {count_subset}")

except Exception as e:
    print(f"\nHa ocurrido un error durante la creación del subset: {e}")

finally:
    con.close()
    print("Conexión a DuckDB cerrada.")
