import duckdb
import pandas as pd
import sys

# --- Configuración ---
REPAIRED_PARQUET_PATH = 'competencia_02_crudo_reparado.parquet'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# ---------------------

print("--- Iniciando Estudio de Variables 'Status' ---")
print(f"Cargando datos desde: {REPAIRED_PARQUET_PATH}\n")

try:
    con = duckdb.connect(database=':memory:', read_only=False)
    
    # Cargar los datos reparados
    con.execute(f"CREATE VIEW data_reparada AS SELECT * FROM read_parquet('{REPAIRED_PARQUET_PATH}')")
    
    # --- ANÁLISIS 1: MATRIZ DE TRANSICIÓN (MASTER) ---
    print("======================================================================")
    print(" Análisis 1.A: Matriz de Transición (Master_status) vs. Clase Ternaria")
    print(" (Qué transiciones de estado se asocian a BAJA+2 o CONTINUA)")
    print("======================================================================")
    
    query_transicion_master = """
    WITH status_con_lag AS (
        SELECT
            numero_de_cliente,
            foto_mes,
            Master_status,
            clase_ternaria, -- <-- ¡NUEVO! Añadimos el target
            -- Obtener el estado del mes anterior para este cliente
            LAG(Master_status, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS prev_Master_status
        FROM data_reparada
    )
    SELECT
        prev_Master_status AS estado_anterior,
        Master_status AS estado_actual,
        clase_ternaria, -- <-- ¡NUEVO! Agrupamos por el target
        COUNT(*) AS total_transiciones
    FROM status_con_lag
    WHERE prev_Master_status IS NOT NULL  -- Excluir el primer mes de cada cliente
    AND Master_status != prev_Master_status -- ¡Solo mostrar cambios reales!
    GROUP BY
        prev_Master_status,
        Master_status,
        clase_ternaria -- <-- ¡NUEVO!
    ORDER BY
        estado_anterior,
        estado_actual,
        total_transiciones DESC;
    """
    
    df_transicion_master = con.execute(query_transicion_master).fetchdf()
    print(df_transicion_master)


    
    # --- ANÁLISIS 1.B: MATRIZ DE TRANSICIÓN (VISA) ---
    print("======================================================================")
    print(" Análisis 1.B: Matriz de Transición (Visa_status) vs. Clase Ternaria")
    print(" (Qué transiciones de estado se asocian a BAJA+2 o CONTINUA)")
    print("======================================================================")
    
    query_transicion_visa = """
    WITH status_con_lag AS (
        SELECT
            numero_de_cliente,
            foto_mes,
            Visa_status,
            clase_ternaria, -- <-- ¡NUEVO! Añadimos el target
            -- Obtener el estado del mes anterior para este cliente
            LAG(Visa_status, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS prev_Visa_status
        FROM data_reparada
    )
    SELECT
        prev_Visa_status AS estado_anterior,
        Visa_status AS estado_actual,
        clase_ternaria, -- <-- ¡NUEVO! Agrupamos por el target
        COUNT(*) AS total_transiciones
    FROM status_con_lag
    WHERE prev_Visa_status IS NOT NULL  -- Excluir el primer mes de cada cliente
    AND Visa_status != prev_Visa_status -- ¡Solo mostrar cambios reales!
    GROUP BY
        prev_Visa_status,
        Visa_status,
        clase_ternaria -- <-- ¡NUEVO!
    ORDER BY
        estado_anterior,
        estado_actual,
        total_transiciones DESC;
    """
    
    df_transicion_visa = con.execute(query_transicion_visa).fetchdf()
    print(df_transicion_visa)


    
    print("\n--- Estudio completado ---")

except Exception as e:
    print(f"\nHa ocurrido un error durante el análisis: {e}")
    if "No such file" in str(e):
        print(f"Error: No se encontró el archivo '{REPAIRED_PARQUET_PATH}'.")
        print("Asegúrate de que el script 'reparacion.py' se haya ejecutado correctamente primero.")

finally:
    con.close()
    print("Conexión a DuckDB cerrada.")
