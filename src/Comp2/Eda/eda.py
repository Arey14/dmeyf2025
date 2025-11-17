# --- 1. Configuración e Importación de Librerías ---

# Instala las librerías necesarias
#!pip install duckdb pandas matplotlib seaborn

# Importaciones
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import json # <-- AÑADIDO: Para guardar el reporte del LLM
from matplotlib.backends.backend_pdf import PdfPages 

# Configuración de visualización
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')
print("Librerías importadas exitosamente.")

# --- 2. Carga de Datos Crudos y Conexión con DuckDB ---

# --- REEMPLAZA ESTO ---
CSV_FILE_PATH = 'competencia_02_crudo.csv' # ¡Corregido en tu script!
# ---------------------

PARQUET_FILE_PATH = CSV_FILE_PATH.rsplit('.', 1)[0] + '.parquet'

con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB (en memoria) establecida.")

try:
    if not os.path.exists(PARQUET_FILE_PATH):
        print(f"Parquet no encontrado. Creando desde '{CSV_FILE_PATH}'...")
        if not os.path.exists(CSV_FILE_PATH):
            raise FileNotFoundError(f"¡El archivo CSV original ({CSV_FILE_PATH}) no se encuentra!")
        
        con.execute(f"""
            COPY (SELECT * FROM read_csv_auto('{CSV_FILE_PATH}')) 
            TO '{PARQUET_FILE_PATH}' (FORMAT 'PARQUET')
        """)
        print(f"Archivo Parquet '{PARQUET_FILE_PATH}' creado.")
    else:
        print(f"Archivo Parquet encontrado: '{PARQUET_FILE_PATH}'.")

    con.execute(f"CREATE VIEW bank_data_raw AS SELECT * FROM read_parquet('{PARQUET_FILE_PATH}')")
    print("Vista 'bank_data_raw' (datos crudos) creada.")

except Exception as e:
    print(f"\nHa ocurrido un error durante la carga de datos: {e}")
    con.close()
    sys.exit(1) # Detener el script si los datos no se pueden cargar

# --- 2.5. Creación de la Variable Target (clase_ternaria) ---

print("\n--- Creando la variable target 'clase_ternaria' ---")

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

query_final_view = """
CREATE OR REPLACE VIEW bank_data_labeled AS
SELECT
    crudo.*,
    clase.clase_ternaria
FROM bank_data_raw AS crudo
INNER JOIN clases_ternarias AS clase
    ON crudo.foto_mes = clase.foto_mes AND crudo.numero_de_cliente = clase.numero_de_cliente;
"""

# Inicializar listas para asegurar que existan (evita NameError)
NUMERIC_COLS = []
CATEGORICAL_COLS = []
ALL_COLUMNS = []

try:
    con.execute(query_clases)
    print("Tabla 'clases_ternarias' (en memoria) creada.")
    
    con.execute(query_final_view)
    print("Vista final 'bank_data_labeled' (datos + target) creada.")
    
    # --- Identificación de Columnas (Sobre la vista final) ---
    column_data = con.execute("DESCRIBE bank_data_labeled").fetchdf()
    ALL_COLUMNS = column_data['column_name'].tolist()
    TARGET_COL = 'clase_ternaria'
    
    NUMERIC_COLS = [
        col for col in ALL_COLUMNS 
        if (
            col.lower().startswith(('m', 'c', 't')) or 
            col.lower().endswith(('_dias', '_edad', '_antiguedad')) or
            col.lower().startswith(('master_', 'visa_'))
           )
        and col != TARGET_COL
    ]
    
    CATEGORICAL_COLS = [
        'active_quarter', 'cliente_vip', 'internet', 'tcallcenter', 
        'thomebanking', 'tmobile_app', 'Master_delinquency', 
        'Master_status', 'Visa_delinquency', 'Visa_status', 
        'ccaja_seguridad'
    ]
    
    NUMERIC_COLS = [col for col in NUMERIC_COLS if col not in CATEGORICAL_COLS and col in ALL_COLUMNS]
    CATEGORICAL_COLS = [col for col in CATEGORICAL_COLS if col in ALL_COLUMNS]
    
    print(f"\nIdentificadas {len(NUMERIC_COLS)} columnas numéricas.")
    print(f"Identificadas {len(CATEGORICAL_COLS)} columnas categóricas.")

except Exception as e:
    print(f"Error al crear la variable target o al describir la tabla: {e}")
    print("ADVERTENCIA: Las listas de columnas están vacías. No se generarán gráficos.")


# --- 3. DEFINICIÓN DE FUNCIONES DE GRÁFICOS ---

# ¡MODIFICADO! Acepta 'eda_data_for_llm'
def plot_monthly_numerical_stats(col_name, pdf_object, eda_data_for_llm):
    """
    Genera gráficos para una variable NUMÉRICA, los guarda en el PDF
    y guarda los datos crudos en el diccionario para el LLM.
    """
    print(f"--- Procesando (Numérica): {col_name} ---")
    
    query = f"""
    SELECT
        foto_mes, 
        COUNT(*) AS total_registros,
        AVG(TRY_CAST({col_name} AS DOUBLE)) AS promedio,
        MEDIAN(TRY_CAST({col_name} AS DOUBLE)) AS mediana,
        MIN(TRY_CAST({col_name} AS DOUBLE)) AS minimo,
        MAX(TRY_CAST({col_name} AS DOUBLE)) AS maximo,
        COUNT(CASE WHEN TRY_CAST({col_name} AS DOUBLE) IS NULL THEN 1 END) AS conteo_nulos,
        COUNT(CASE WHEN TRY_CAST({col_name} AS DOUBLE) = 0 THEN 1 END) AS conteo_ceros
    FROM bank_data_labeled
    GROUP BY foto_mes
    ORDER BY foto_mes
    """

    try:
        df_stats = con.execute(query).fetchdf()
        if df_stats.empty:
            print(f"No se pudieron obtener datos para {col_name}")
            return

        # --- ¡NUEVO! Guardar datos para el LLM ---
        # Se guarda antes de convertir foto_mes a datetime para el plot
        if eda_data_for_llm is not None:
             eda_data_for_llm["eda_summary"][col_name] = {
                "type": "numerical",
                "monthly_stats": df_stats.to_dict('records')
            }
        # ----------------------------------------

        df_stats['foto_mes'] = pd.to_datetime(df_stats['foto_mes'], format='%Y%m')

        fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
        fig.suptitle(f'Evolución Mensual de: {col_name}', fontsize=18, y=0.98)

        sns.lineplot(data=df_stats, x='foto_mes', y='promedio', ax=axes[0], label='Promedio', marker='o')
        sns.lineplot(data=df_stats, x='foto_mes', y='mediana', ax=axes[0], label='Mediana', marker='x', linestyle='--')
        axes[0].set_title('Promedio y Mediana'); axes[0].legend()
        
        sns.lineplot(data=df_stats, x='foto_mes', y='minimo', ax=axes[1], label='Mínimo', marker='o')
        sns.lineplot(data=df_stats, x='foto_mes', y='maximo', ax=axes[1], label='Máximo', marker='o')
        axes[1].set_title('Mínimo y Máximo'); axes[1].legend()

        sns.lineplot(data=df_stats, x='foto_mes', y='conteo_nulos', ax=axes[2], label='Conteo de Nulos', marker='o', color='red')
        axes[2].set_title('Conteo de Nulos'); axes[2].legend()

        sns.lineplot(data=df_stats, x='foto_mes', y='conteo_ceros', ax=axes[3], label='Conteo de Ceros', marker='o', color='purple')
        axes[3].set_title('Conteo de Ceros'); axes[3].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        pdf_object.savefig(fig)
        plt.close(fig) 

    except Exception as e:
        print(f"Error al procesar {col_name}: {e}")

# ¡MODIFICADO! Acepta 'eda_data_for_llm'
def plot_monthly_categorical_stats(col_name, pdf_object, eda_data_for_llm):
    """
    Genera gráficos para una variable CATEGÓRICA, los guarda en el PDF
    y guarda los datos crudos en el diccionario para el LLM.
    """
    print(f"--- Procesando (Categórica): {col_name} ---")

    query_dist = f"""
    SELECT foto_mes, {col_name}, COUNT(*) AS conteo
    FROM bank_data_labeled
    GROUP BY foto_mes, {col_name}
    ORDER BY foto_mes, {col_name}
    """
    query_nulls = f"""
    SELECT foto_mes, COUNT(CASE WHEN {col_name} IS NULL THEN 1 END) AS conteo_nulos
    FROM bank_data_labeled
    GROUP BY foto_mes
    ORDER BY foto_mes
    """
    
    try:
        df_dist = con.execute(query_dist).fetchdf()
        df_nulls = con.execute(query_nulls).fetchdf()
        
        if df_dist.empty:
            print(f"No se pudieron obtener datos para {col_name}")
            return
            
        # Calcular proporciones para el LLM
        df_total_mes = df_dist.groupby('foto_mes')['conteo'].sum().reset_index().rename(columns={'conteo': 'total_mes'})
        df_dist_llm = df_dist.merge(df_total_mes, on='foto_mes')
        df_dist_llm['proporcion'] = df_dist_llm['conteo'] / df_dist_llm['total_mes']

        # --- ¡NUEVO! Guardar datos para el LLM ---
        if eda_data_for_llm is not None:
             eda_data_for_llm["eda_summary"][col_name] = {
                "type": "categorical",
                "monthly_distribution": df_dist_llm.to_dict('records'),
                "monthly_nulls": df_nulls.to_dict('records')
            }
        # ----------------------------------------
            
        # Convertir a datetime para graficar
        df_dist['foto_mes'] = pd.to_datetime(df_dist['foto_mes'], format='%Y%m')
        df_nulls['foto_mes'] = pd.to_datetime(df_nulls['foto_mes'], format='%Y%m')
        
        # Necesitamos recalcular proporciones con el df de datetime para el plot
        df_total_mes_plot = df_dist.groupby('foto_mes')['conteo'].sum().reset_index().rename(columns={'conteo': 'total_mes'})
        df_dist_plot = df_dist.merge(df_total_mes_plot, on='foto_mes')
        df_dist_plot['proporcion'] = df_dist_plot['conteo'] / df_dist_plot['total_mes']

        fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f'Evolución Mensual de: {col_name}', fontsize=18, y=0.98)

        # Gráfico 1: Distribución de Proporciones
        df_pivot = df_dist_plot.pivot(index='foto_mes', columns=col_name, values='proporcion').fillna(0)
        df_pivot.plot(kind='bar', stacked=True, ax=axes[0], width=0.8)
        axes[0].set_title('Distribución de Valores (Proporción)')
        axes[0].set_ylabel('Proporción')
        axes[0].legend(title=col_name, bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)

        # Gráfico 2: Conteo de Nulos
        sns.lineplot(data=df_nulls, x='foto_mes', y='conteo_nulos', ax=axes[1], label='Conteo de Nulos', marker='o', color='red')
        axes[1].set_title('Conteo de Nulos'); axes[1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        pdf_object.savefig(fig)
        plt.close(fig)

    except Exception as e:
        print(f"Error al procesar {col_name}: {e}")

# --- 3.5. Ejecución del Análisis de Evolución (Guardado en PDF y JSON) ---

# Inicializar contenedores de salida
pdf_pages = None 
json_report_filename = 'eda_reporte_llm.json'
eda_data_for_llm = {
    "eda_summary": {},
    "data_quality_checks": {
        "count_vs_amount_inconsistencies": [],
        "unexpected_categorical_values": [],
        "other_logical_inconsistencies": []
    }
}
# -----------------------------------------------

pdf_report_filename = 'eda_reporte_completo.pdf'
try:
    pdf_pages = PdfPages(pdf_report_filename)
    print(f"\n--- Creando reporte PDF: {pdf_report_filename} ---")
    print(f"--- Preparando datos para reporte JSON: {json_report_filename} ---")

    print("\n### Iniciando Análisis de TODAS las Variables Numéricas ###")
    for col in NUMERIC_COLS:
        # ¡MODIFICADO! Se pasa el dict del LLM
        plot_monthly_numerical_stats(col, pdf_pages, eda_data_for_llm)

    print("\n### Iniciando Análisis de TODAS las Variables Categóricas ###")
    for col in CATEGORICAL_COLS:
        # ¡MODIFICADO! Se pasa el dict del LLM
        plot_monthly_categorical_stats(col, pdf_pages, eda_data_for_llm)

    print("Procesamiento de variables finalizado.")

except Exception as e:
    print(f"\n¡¡Error CRÍTICO durante la generación de gráficos!!: {e}")
    print("El PDF puede estar incompleto o vacío.")


# --- 4. Análisis Específico de Anomalías ---

# A. Análisis del Target (clase_ternaria)
print("\n### A. Guardando gráfico del Target en PDF y JSON ###")
try:
    query_target = "SELECT foto_mes, clase_ternaria, COUNT(*) AS conteo FROM bank_data_labeled GROUP BY foto_mes, clase_ternaria ORDER BY foto_mes, clase_ternaria"
    df_target = con.execute(query_target).fetchdf()

    # --- ¡NUEVO! Guardar datos para el LLM ---
    eda_data_for_llm["eda_summary"]["clase_ternaria"] = {
        "type": "target",
        "monthly_distribution": df_target.to_dict('records')
    }
    # ----------------------------------------
    
    df_target['foto_mes'] = pd.to_datetime(df_target['foto_mes'], format='%Y%m')
    df_target_pivot = df_target.pivot(index='foto_mes', columns='clase_ternaria', values='conteo').fillna(0)
    
    df_target_props = df_target_pivot.divide(df_target_pivot.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    df_target_props.plot(kind='bar', stacked=True, ax=ax)
    
    fig.suptitle('Distribución del Target (clase_ternaria) por Mes', fontsize=18, y=0.98)
    ax.set_ylabel('Proporción de Clientes')
    ax.legend(title='Clase', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if pdf_pages:
        pdf_pages.savefig(fig)
        print("Gráfico del Target guardado.")
    else:
        print("No se pudo guardar el gráfico del Target (PDF no inicializado).")
    plt.close(fig)

except Exception as e:
    print(f"Error analizando el target: {e}")

# --- B. Verificación de Lógica de Negocio (Impreso en Consola y guardado en JSON) ---
print("\n### B. Verificación de Lógica de Negocio (Sistemática) ###")
    
print("\n--- Check 1: Inconsistencias Conteo (c) vs. Monto (m) ---")

check_pairs = [
    ('ctarjeta_debito_transacciones', 'mautoservicio'), 
    ('ctarjeta_visa_transacciones', 'mtarjeta_visa_consumo'), 
    ('ctarjeta_master_transacciones', 'mtarjeta_master_consumo'), 
    ('cprestamos_personales', 'mprestamos_personales'), 
    ('cprestamos_prendarios', 'mprestamos_prendarios'), 
    ('cprestamos_hipotecarios', 'mprestamos_hipotecarios'), 
    ('cpayroll_trx', 'mpayroll'), 
    ('ccuenta_debitos_automaticos', 'mcuenta_debitos_automaticos'), 
    ('ctarjeta_visa_debitos_automaticos', 'mtarjeta_visa_debitos_automaticos'), 
    ('ctarjeta_master_debitos_automaticos', 'mttarjeta_master_debitos_automaticos'), 
    ('cpagodeservicios', 'mpagodeserviciospesos'), 
    ('cpagomiscuentas', 'mpagomiscuentaspesos'), 
    ('ccajeros_propios_descuentos', 'mcajeros_propios_descuentos'), 
    ('ctarjeta_visa_descuentos', 'mtarjeta_visa_descuentos'), 
    ('ctarjeta_master_descuentos', 'mtarjeta_master_descuentos'), 
    ('cforex_buy', 'mforex_buypesos'), 
    ('cforex_sell', 'mforex_sellpesos'), 
    ('ctransferencias_recibidas', 'mtransferencias_recibidaspesos'), 
    ('ctransferencias_emitidas', 'mtransferencias_emitidaspesos'), 
    ('cextraccion_autoservicio', 'mextraccion_autoserviciopesos'), 
    ('ccheques_depositados', 'mcheques_depositadospesos'), 
    ('ccheques_emitidos', 'mcheques_emitidospesos'), 
    ('ccheques_depositados_rechazados', 'mcheques_depositados_rechazadospesos'), 
    ('ccheques_emitidos_rechazados', 'mcheques_emitidos_rechazadospesos'), 
    ('catm_trx', 'matmpesos'), 
    ('catm_trx_other', 'matm_otherpesos') 
]

validated_pairs = []
for c_col, m_col in check_pairs:
    if c_col in ALL_COLUMNS and m_col in ALL_COLUMNS:
        validated_pairs.append((c_col, m_col))

print(f"Validando {len(validated_pairs)} pares de Conteo/Monto...")
inconsistencias_encontradas = 0

for c_col, m_col in validated_pairs:
    query = f"""
    SELECT foto_mes, COUNT(*) AS inconsistencias
    FROM bank_data_labeled
    WHERE ({c_col} = 0 OR {c_col} IS NULL) AND TRY_CAST({m_col} AS DOUBLE) > 0
    GROUP BY foto_mes
    HAVING inconsistencias > 0
    """
    try:
        df_check = con.execute(query).fetchdf()
        if not df_check.empty:
            print(f"\n¡Inconsistencia encontrada!: {c_col} vs {m_col}")
            print(df_check)
            inconsistencias_encontradas += 1
            
            # --- ¡NUEVO! Guardar anomalía para el LLM ---
            for record in df_check.to_dict('records'):
                eda_data_for_llm["data_quality_checks"]["count_vs_amount_inconsistencies"].append({
                    "check": f"{c_col} vs {m_col}",
                    "foto_mes": record['foto_mes'],
                    "inconsistencias": record['inconsistencias']
                })
            # -------------------------------------------
            
    except Exception as e:
        print(f"Error validando par ({c_col}, {m_col}): {e}")

if inconsistencias_encontradas == 0:
    print("Check 1: OK - No se encontraron inconsistencias de Conteo vs Monto.")


# --- Check 2: Columnas con Valores Definidos ---
print("\n--- Check 2: Valores inesperados en columnas categóricas ---")

defined_values_checks = {
    'ccaja_seguridad': [0, 1], 
    'Master_delinquency': [0, 1], 
    'Visa_delinquency': [0, 1], 
    'Master_status': [0, 6, 7, 9], 
    'Visa_status': [0, 6, 7, 9] 
}
valores_inesperados = 0

for col, values in defined_values_checks.items():
    if col not in CATEGORICAL_COLS:
        continue
        
    values_str = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in values])
    query = f"""
    SELECT DISTINCT {col}
    FROM bank_data_labeled
    WHERE {col} NOT IN ({values_str}) AND {col} IS NOT NULL
    """
    try:
        df_unexpected = con.execute(query).fetchdf()
        if not df_unexpected.empty:
            unexpected_list = df_unexpected[col].tolist()
            print(f"\n¡Valores inesperados encontrados en '{col}'! (Esperados: {values_str})")
            print(unexpected_list)
            valores_inesperados += 1

            # --- ¡NUEVO! Guardar anomalía para el LLM ---
            for val in unexpected_list:
                eda_data_for_llm["data_quality_checks"]["unexpected_categorical_values"].append({
                    "column": col,
                    "unexpected_value": val
                })
            # -------------------------------------------
            
    except Exception as e:
        print(f"Error validando valores definidos para '{col}': {e}")

if valores_inesperados == 0:
    print("Check 2: OK - No se encontraron valores categóricos inesperados.")


# --- 5. Cierre de Conexión y PDF ---

# 1. Cerrar el PDF
if pdf_pages:
    try:
        pdf_pages.close()
        print(f"\n--- Reporte PDF guardado exitosamente: {pdf_report_filename} ---")
    except Exception as e:
        print(f"Error al guardar el PDF: {e}")
else:
    print("\n--- No se generó ningún reporte PDF debido a errores previos. ---")

# 2. Guardar el JSON para el LLM
try:
    with open(json_report_filename, 'w', encoding='utf-8') as f:
        # Usamos indent=4 para que sea legible por humanos también
        json.dump(eda_data_for_llm, f, indent=4, ensure_ascii=False)
    print(f"\n--- Reporte JSON para LLM guardado exitosamente: {json_report_filename} ---")
except Exception as e:
    print(f"Error al guardar el JSON: {e}")

# 3. Cerrar la conexión a la base de datos
con.close()
print("\n--- EDA Exhaustivo finalizado. Conexión a DuckDB cerrada. ---")
