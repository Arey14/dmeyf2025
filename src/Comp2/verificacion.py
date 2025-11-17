import duckdb
import os
import sys
import pandas as pd

# --- Configuración de Archivos ---
# Apunta al archivo que generó el script de reparación
REPAIRED_PARQUET_PATH = 'competencia_02_crudo_reparado.parquet'
# ---------------------------------

print(f"--- Iniciando Verificación de Datos Reparados ---")
print(f"Cargando archivo: {REPAIRED_PARQUET_PATH}\n")

if not os.path.exists(REPAIRED_PARQUET_PATH):
    print(f"Error: No se encontró el archivo reparado '{REPAIRED_PARQUET_PATH}'.")
    print("Asegúrate de que 'reparacion.py' se haya ejecutado exitosamente.")
    sys.exit(1)

# Conectar a DuckDB en memoria
con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB establecida.")

try:
    # Cargar los datos reparados en una vista para consultar
    con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{REPAIRED_PARQUET_PATH}')")
    print("Vista 'data' (datos reparados) creada.\n")
    print("--- Ejecutando Pruebas de Calidad de Datos ---\n")

    # --- 1. Verificación de Exclusión de Mes ---
    print("1. Verificando Exclusión de Mes Anómalo (202006):")
    count_202006 = con.execute("SELECT COUNT(*) FROM data WHERE foto_mes = 202006").fetchone()[0]
    if count_202006 == 0:
        print("   [OK] El mes 202006 ha sido excluido correctamente.\n")
    else:
        print(f"   [FALLO] Se encontraron {count_202006} registros del mes 202006. No se excluyó correctamente.\n")

    # --- 2. Verificación de Interpolación ---
    print("2. Verificando Interpolación de Gaps (201905, 201910):")
    # Verificamos 'mrentabilidad' en 201905. El original era 0.0, el interpolado no debería serlo.
    avg_rent_201905 = con.execute("SELECT AVG(mrentabilidad) FROM data WHERE foto_mes = 201905").fetchone()[0]
    if avg_rent_201905 != 0.0:
        print(f"   [OK] Mes 201905 'mrentabilidad' fue imputado (Promedio: {avg_rent_201905:.2f}).")
    else:
        print(f"   [FALLO] Mes 201905 'mrentabilidad' sigue siendo 0.0.")
    
    # Verificamos 'mcomisiones_otras' en 201910.
    avg_com_201910 = con.execute("SELECT AVG(mcomisiones_otras) FROM data WHERE foto_mes = 201910").fetchone()[0]
    if avg_com_201910 != 0.0:
        print(f"   [OK] Mes 201910 'mcomisiones_otras' fue imputado (Promedio: {avg_com_201910:.2f}).\n")
    else:
        print(f"   [FALLO] Mes 201910 'mcomisiones_otras' sigue siendo 0.0.\n")

    # --- 3. Verificación de Limpieza de Anomalías ---
    print("3. Verificando Limpieza de Anomalías:")
    
    # Check 'ccaja_seguridad'
    max_caja = con.execute("SELECT MAX(ccaja_seguridad) FROM data").fetchone()[0]
    if max_caja == 1:
        print(f"   [OK] 'ccaja_seguridad' limpiada (Max: {max_caja}).")
    else:
        print(f"   [FALLO] 'ccaja_seguridad' contiene valores > 1 (Max: {max_caja}).")

    # Check 'cliente_edad'
    edad_anomalos = con.execute("SELECT COUNT(*) FROM data WHERE cliente_edad < 18 OR cliente_edad > 100").fetchone()[0]
    if edad_anomalos == 0:
        print(f"   [OK] 'cliente_edad' limpiada (0 registros fuera de rango).")
    else:
        print(f"   [FALLO] 'cliente_edad' todavía contiene {edad_anomalos} registros fuera de rango (18-100).")

    # Check 'Fvencimiento'
    fechas_anomalas = con.execute("SELECT COUNT(*) FROM data WHERE Master_Fvencimiento < -20000 OR Visa_Fvencimiento < -20000").fetchone()[0]
    if fechas_anomalas == 0:
        print(f"   [OK] Fechas 'Fvencimiento' limpiadas (0 registros anómalos).\n")
    else:
        print(f"   [FALLO] Fechas 'Fvencimiento' todavía contienen {fechas_anomalas} registros anómalos.\n")

    # --- 4. Verificación de Inconsistencias Lógicas ---
    print("4. Verificando Inconsistencias Lógicas:")
    
    # Check 'cprestamos_prendarios'
    prest_prend_incons = con.execute("SELECT COUNT(*) FROM data WHERE cprestamos_prendarios = 0 AND mprestamos_prendarios > 0").fetchone()[0]
    if prest_prend_incons == 0:
        print(f"   [OK] 'cprestamos_prendarios' corregido (0 inconsistencias lógicas).")
    else:
        print(f"   [FALLO] 'cprestamos_prendarios' todavía tiene {prest_prend_incons} inconsistencias.")

    # Check 'cpayroll_trx'
    payroll_incons = con.execute("SELECT COUNT(*) FROM data WHERE cpayroll_trx = 0 AND mpayroll > 0").fetchone()[0]
    if payroll_incons == 0:
        print(f"   [OK] 'cpayroll_trx' corregido (0 inconsistencias lógicas).\n")
    else:
        print(f"   [FALLO] 'cpayroll_trx' todavía tiene {payroll_incons} inconsistencias.\n")

    # --- 5. Verificación de Concept Drift ---
    print("5. Verificando Manejo de Concept Drift (Columnas Nuevas):")
    
    column_data = con.execute("DESCRIBE data").fetchdf()
    columns = column_data['column_name'].tolist()
    
    if 'cmobile_app_trx' not in columns:
        print(f"   [OK] Columna original 'cmobile_app_trx' eliminada.")
    else:
        print(f"   [FALLO] Columna original 'cmobile_app_trx' todavía existe.")

    if 'cmobile_app_trx_pre202010' in columns:
        print(f"   [OK] Nueva columna 'cmobile_app_trx_pre202010' fue creada.")
    else:
        print(f"   [FALLO] Nueva columna 'cmobile_app_trx_pre202010' NO fue creada.")

    if 'cmobile_app_trx_post202010' in columns:
        print(f"   [OK] Nueva columna 'cmobile_app_trx_post202010' fue creada.")
    else:
        print(f"   [FALLO] Nueva columna 'cmobile_app_trx_post202010' NO fue creada.")
    
    # Check tmobile_app NULLs
    nulls_tmobile = con.execute("SELECT COUNT(*) FROM data WHERE foto_mes < 201907 AND tmobile_app IS NOT NULL").fetchone()[0]
    if nulls_tmobile == 0:
        print(f"   [OK] 'tmobile_app' se mantuvo como NULL antes de 201907.\n")
    else:
        print(f"   [FALLO] 'tmobile_app' tiene valores no nulos antes de 201907.\n")


except Exception as e:
    print(f"\nHa ocurrido un error durante el proceso de verificación: {e}")

finally:
    con.close()
    print("--- Verificación Finalizada. Conexión a DuckDB cerrada. ---")
