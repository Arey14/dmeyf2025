import pandas as pd
import os
import glob
import json

# --- Configuración ---
# Esto buscará todos los archivos que hemos creado, como:
# - feature_importances_avg.csv
# - feature_importances_cv.csv
# - feature_boruta_selection.csv (si lo generaste)
FILES_TO_CONVERT = glob.glob('feature_*.csv')
# ---------------------

print(f"--- Iniciando Conversión de CSV a JSON ---")

if not FILES_TO_CONVERT:
    print("Error: No se encontraron archivos CSV que comiencen con 'feature_'.")
    print("Asegúrate de que los scripts de selección de features se hayan ejecutado.")
    sys.exit(1)

print(f"Archivos encontrados para convertir: {FILES_TO_CONVERT}\n")

for csv_file in FILES_TO_CONVERT:
    try:
        # 1. Generar el nuevo nombre de archivo
        json_file = csv_file.replace('.csv', '.json')
        
        # 2. Leer el CSV
        print(f"Leyendo '{csv_file}'...")
        df = pd.read_csv(csv_file)
        
        # 3. Guardar como JSON
        #    orient='records' es el formato ideal para análisis (lista de diccionarios)
        #    indent=4 lo hace legible para humanos
        df.to_json(json_file, orient='records', indent=4, force_ascii=False)
        
        print(f"   [ÉXITO] Archivo '{json_file}' guardado correctamente.\n")

    except Exception as e:
        print(f"   [FALLO] No se pudo convertir '{csv_file}': {e}\n")

print("--- Conversión completada ---")
