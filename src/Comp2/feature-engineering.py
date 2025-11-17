import duckdb
import os
import sys
import pandas as pd
import time

# --- 1. CONFIGURACIÓN MODULAR ---
# True = Crear este bloque de features.
# False = Omitir este bloque de features.
FE_CONFIG = {
    "crear_combinaciones_basicas": True,
    "crear_deltas_y_tendencias": True,
    "crear_tendencias_lineales": True, # NUEVO: Para la pendiente (slope)
    "crear_features_categoricas": True,
    "crear_ratios_avanzados": True,
    "crear_rankings_cero_fijo": True, # MODIFICADO: Renombrada la función
}

# Define las ventanas de tiempo para deltas y tendencias
TIME_WINDOWS = [1, 3, 6]
# ---------------------------------


# --- Configuración de Archivos ---
REPAIRED_PARQUET_PATH = 'competencia_02_crudo_reparado.parquet'
FEATURES_PARQUET_PATH = 'features_dataset.parquet'
# ---------------------------------


def crear_combinaciones_basicas():
    """
    (Refactorizado - Patrón Robusto)
    Crea nuevas variables a partir de la combinación de columnas existentes.
    
    MODIFICADO:
    - Añadidas las combinaciones Visa+Mastercard de 'full_pipeline.txt'
    - Se usa COALESCE(col, 0) para replicar el fillna(0) de Python antes de sumar.
    """
    expressions = [
        # Features de Estacionalidad
        "(foto_mes / 100)::INTEGER AS anio",
        "(foto_mes % 100) AS mes",
        
        # Suma de todos los consumos de tarjetas
        "(mtarjeta_visa_consumo + mtarjeta_master_consumo) AS mconsumo_total_tarjetas",
        
        # (NUEVO) Combinaciones Visa+Mastercard (de full_pipeline.txt)
        "(COALESCE(Master_msaldototal, 0) + COALESCE(Visa_msaldototal, 0)) AS vm_msaldototal",
        "(COALESCE(Master_mconsumospesos, 0) + COALESCE(Visa_mconsumospesos, 0)) AS vm_mconsumospesos",
        "(COALESCE(Master_mlimitecompra, 0) + COALESCE(Visa_mlimitecompra, 0)) AS vm_mlimitecompra",
        "(COALESCE(Master_madelantopesos, 0) + COALESCE(Visa_madelantopesos, 0)) AS vm_madelantopesos",
        "(COALESCE(Master_mpagado, 0) + COALESCE(Visa_mpagado, 0)) AS vm_mpagado",
        
        # Total de productos de inversión (Plazos Fijos + Inversiones 1 y 2)
        "(cplazo_fijo + cinversion1 + cinversion2) AS cproductos_inversion",
        
        # Total de seguros
        "(cseguro_vida + cseguro_auto + cseguro_vivienda + cseguro_accidentes_personales) AS cproductos_seguros",
        
        # Total de préstamos
        "(cprestamos_personales + cprestamos_prendarios + cprestamos_hipotecarios) AS cproductos_prestamos",

        # Total de "Anclaje" (Débitos Automáticos)
        "(ccuenta_debitos_automaticos + ctarjeta_visa_debitos_automaticos + ctarjeta_master_debitos_automaticos) AS ctotal_debitos_automaticos",
        "(mcuenta_debitos_automaticos + mttarjeta_visa_debitos_automaticos + mttarjeta_master_debitos_automaticos) AS mtotal_debitos_automaticos",

        # Total de "Dolor" (Cheques Rechazados)
        "(ccheques_depositados_rechazados + ccheques_emitidos_rechazados) AS ctotal_cheques_rechazados",
        "(mcheques_depositados_rechazados + mcheques_emitidos_rechazados) AS mtotal_cheques_rechazados",
        
        # Peor estado de Tarjetas (0=OK, 9=Cerrada)
        "GREATEST(Master_status, Visa_status) AS peor_estado_tarjetas"
    ]
    
    if not expressions:
        return None
        
    body = ",\n".join(expressions)
    return f"\n-- --- 1. Combinaciones Básicas (Ratios y Sumas) ---\n{body}"


def crear_deltas_y_tendencias(cols_to_process, windows=[1, 3, 6]):
    """
    (Refactorizado - Patrón Robusto)
    Crea deltas (lags), promedios móviles, y otras tendencias para variables numéricas.
    
    MODIFICADO:
    - 'cols_to_process' ahora es un parámetro.
    - Se aplica '::DOUBLE' a todas las columnas para manejar de forma segura
      los tipos VARCHAR que contienen números.
    """
    
    expressions = []
    
    if not cols_to_process or not windows:
        print("Advertencia: No se generarán deltas/tendencias. 'cols_to_process' o 'windows' están vacíos.")
    else:
        max_w = max(windows)
        for col in cols_to_process:
            # CASTING SEGURO a ::DOUBLE
            col_casted = f'"{col}"::DOUBLE' # Usar comillas por si el nombre tiene mayúsculas
            
            for w in windows:
                # DELTAS (Lag N)
                expressions.append(f"({col_casted} - LAG({col_casted}, {w}) OVER w_cliente) AS delta_{col}_{w}m")
                
                # TENDENCIAS (Promedio Móvil N)
                expressions.append(f"AVG({col_casted}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW) AS avg_{col}_{w}m")

            # AGREGADOS (Max, Sum en la ventana más grande)
            expressions.append(f"MAX({col_casted}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {max_w-1} PRECEDING AND CURRENT ROW) AS max_{col}_{max_w}m")
            expressions.append(f"SUM({col_casted}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {max_w-1} PRECEDING AND CURRENT ROW) AS sum_{col}_{max_w}m")
            
            # RATIOS vs. Tendencia
            expressions.append(f"{col_casted} / NULLIF(AVG({col_casted}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {max_w-1} PRECEDING AND CURRENT ROW), 0) AS ratio_{col}_vs_avg{max_w}m")

    # Features de Volatilidad (StdDev) - Se añaden siempre
    # MODIFICADO: Añadido ::DOUBLE
    expressions.append("STDDEV_SAMP(mcuentas_saldo::DOUBLE) OVER w_cliente_3m AS stddev_mcuentas_saldo_3m")
    expressions.append("STDDEV_SAMP(mcuentas_saldo::DOUBLE) OVER w_cliente_6m AS stddev_mcuentas_saldo_6m") 

    # Features de Frecuencia de "Dolor" - Se añaden siempre
    # MODIFICADO: Añadido ::DOUBLE
    expressions.append("SUM(CASE WHEN ctrx_quarter::DOUBLE = 0 THEN 1 ELSE 0 END) OVER w_cliente_6m AS count_meses_sin_trx_6m")
    expressions.append("SUM(CASE WHEN mtarjeta_visa_consumo::DOUBLE = 0 THEN 1 ELSE 0 END) OVER w_cliente_6m AS count_meses_sin_consumo_visa_6m")
    expressions.append("SUM(CASE WHEN mcuentas_saldo::DOUBLE < 0 THEN 1 ELSE 0 END) OVER w_cliente_6m AS count_meses_descubierto_6m")
    expressions.append("SUM(Master_delinquency::DOUBLE) OVER w_cliente_6m AS count_meses_mora_master_6m")
    expressions.append("SUM(Visa_delinquency::DOUBLE) OVER w_cliente_6m AS count_meses_mora_visa_6m")
    expressions.append("SUM(ctotal_cheques_rechazados::DOUBLE) OVER w_cliente_6m AS sum_cheques_rechazados_6m") 

    if not expressions:
        return None
        
    body = ",\n".join(expressions)
    return f"\n-- --- 2. Deltas y Tendencias (Ventanas Móviles) ---\n{body}"


def crear_tendencias_lineales():
    """
    (NUEVO)
    Calcula la pendiente de la tendencia (regresión lineal) para un
    conjunto específico de variables clave, imitando 'np.polyfit'
    de full_pipeline.txt usando REGR_SLOPE de SQL.
    """
    
    # Lista de columnas base (inspirada en full_pipeline.txt)
    # Asegúrate que 'mrentabilidad_anual' exista en el parquet
    cols_for_trend = [
        'mrentabilidad',
        'mrentabilidad_annual', 
        'ctrx_quarter',
        'cproductos'
    ]
    
    windows = [3, 6] # Usar 3m (como en Python) y 6m
    expressions = []
    
    for col in cols_for_trend:
        col_casted = f'"{col}"::DOUBLE'
        for w in windows:
            window_name = f'w_cliente_{w}m' # Usar las ventanas ya definidas
            expressions.append(f"REGR_SLOPE({col_casted}, foto_mes) OVER {window_name} AS tend_{col}_{w}m")

    # Caso especial para 'vm_msaldototal' (alias de combinaciones básicas)
    # Re-calculamos la suma inline para que funcione en la misma query
    col_alias = "(COALESCE(Master_msaldototal, 0) + COALESCE(Visa_msaldototal, 0))"
    for w in windows:
        window_name = f'w_cliente_{w}m'
        expressions.append(f"REGR_SLOPE({col_alias}, foto_mes) OVER {window_name} AS tend_vm_msaldototal_{w}m")
    
    if not expressions:
        return None

    body = ",\n".join(expressions)
    return f"\n-- --- 2b. Tendencias (Regresión Lineal) ---\n{body}"


def crear_features_categoricas():
    """
    (Refactorizado - Patrón Robusto)
    Crea features que detectan cambios de estado en variables categóricas.
    (Esta función mantiene su propia lógica y lista de columnas,
     ya que su propósito es diferente a los deltas numéricos)
    """
    
    # Lista de variables categóricas 0/1 para rastrear cambios
    cols_to_track_simple = [
        'active_quarter',
        'cliente_vip',
        'internet',
        'tcallcenter',
        'thomebanking',
        'tmobile_app', # Esta columna era VARCHAR
        'Master_delinquency',
        'Visa_delinquency',
        'ccaja_seguridad',
        'ctotal_debitos_automaticos', 
        'ctotal_cheques_rechazados'   
    ]
    
    expressions = []
    
    for col in cols_to_track_simple:
        # Se añade '::INTEGER' a {col} para forzar la conversión de VARCHAR
        
        # 1. Delta Categórico: ¿Cambió el estado este mes?
        expressions.append(f"""
        CASE 
            WHEN "{col}"::INTEGER != LAG("{col}"::INTEGER) OVER w_cliente THEN 1
            WHEN LAG("{col}"::INTEGER) OVER w_cliente IS NULL AND "{col}"::INTEGER IS NOT NULL THEN 1 -- Primer mes
            ELSE 0 
        END AS cambio_{col}
        """)
        
        # 2. Caída de Producto: ¿Pasó de >0 (activo) a 0 (inactivo)?
        expressions.append(f"""
        CASE 
            WHEN "{col}"::INTEGER = 0 AND LAG("{col}"::INTEGER) OVER w_cliente > 0 THEN 1
            ELSE 0 
        END AS dejo_{col}
        """)
        
        # 3. (NUEVO) Inicio de Evento: ¿Pasó de 0 a >0?
        expressions.append(f"""
        CASE 
            WHEN "{col}"::INTEGER > 0 AND LAG("{col}"::INTEGER) OVER w_cliente = 0 THEN 1
            ELSE 0 
        END AS inicio_{col}
        """)
            
    # Lógica Específica para Master_status y Visa_status
    status_cols_to_track = ['Master_status', 'Visa_status', 'peor_estado_tarjetas']
    
    for col in status_cols_to_track:
        # Evento: Inicia proceso de cierre (pasa de 0 a 6, 7, o 9)
        expressions.append(f"""
        CASE 
            WHEN "{col}" > 0 AND LAG("{col}") OVER w_cliente = 0 THEN 1
            ELSE 0 
        END AS inicio_cierre_{col}
        """)
        
        # Evento: Cuenta se cierra (llega a 9)
        expressions.append(f"""
        CASE 
            WHEN "{col}" = 9 AND LAG("{col}") OVER w_cliente != 9 THEN 1
            ELSE 0 
        END AS cuenta_cerrada_{col}
        """)

        # Evento: Reactivación de cuenta (vuelve de 9 a 0)
        expressions.append(f"""
        CASE 
            WHEN "{col}" = 0 AND LAG("{col}") OVER w_cliente = 9 THEN 1
            ELSE 0 
        END AS reactivacion_cuenta_{col}
        """)

    if not expressions:
        return None

    body = ",\n".join(expressions)
    return f"\n-- --- 3. Features Categóricas (Cambios de Estado) ---\n{body}"


def crear_ratios_avanzados(cols_for_drift_ratios, windows=[1, 3, 6]):
    """
    (Refactorizado V4 - CORRECCIÓN VARCHAR)
    Crea un cuarto paso modular para ratios avanzados.
    
    MODIFICADO:
    - 'cols_for_drift_ratios' ahora es un parámetro para la sección de Drift.
    - Se aplica '::DOUBLE' en la sección de Drift.
    - Añadidos ratios Visa+Mastercard de 'full_pipeline.txt'.
    """

    # --- 1. Definir Ratios de Composición (Estos son fijos) ---
    composition_expressions = [
        "mcomisiones / NULLIF(mrentabilidad, 0) AS ratio_comisiones_rentabilidad",
        "mcuentas_saldo / NULLIF((Master_mlimitecompra + Visa_mlimitecompra), 0) AS ratio_saldo_limite",
        
        # --- CORRECCIÓN ---
        # Se añade '::INTEGER' a 'cmobile_app_trx_pre202010' porque es VARCHAR
        "chomebanking_transacciones / NULLIF(cmobile_app_trx_pre202010::INTEGER, 0) AS ratio_hb_vs_app_pre202010",
        # --- FIN CORRECCIÓN ---
        
        # (NUEVO) Ratios Visa+Mastercard (de full_pipeline.txt)
        # Re-calculamos las sumas con COALESCE para evitar usar alias de la misma query
        "(COALESCE(Master_msaldototal, 0) + COALESCE(Visa_msaldototal, 0)) / NULLIF((COALESCE(Master_mlimitecompra, 0) + COALESCE(Visa_mlimitecompra, 0)), 0) AS vmr_msaldototal_div_mlimitecompra",
        "(COALESCE(Master_mconsumospesos, 0) + COALESCE(Visa_mconsumospesos, 0)) / NULLIF((COALESCE(Master_mlimitecompra, 0) + COALESCE(Visa_mlimitecompra, 0)), 0) AS vmr_mconsumospesos_div_mlimitecompra"
    ]
    
    # --- 2. Generar Ratios de Drift (Ahora dinámicos) ---
    drift_expressions = []
    
    # MODIFICADO: Usar la lista de columnas dinámica
    cols_to_ratio = cols_for_drift_ratios

    # Solo generar si hay columnas Y windows
    if cols_to_ratio and windows:
        for col in cols_to_ratio:
            # CASTING SEGURO a ::DOUBLE
            col_casted = f'"{col}"::DOUBLE'
            
            for w in windows:
                expr = f"{col_casted} / NULLIF(LAG({col_casted}, {w}) OVER w_cliente, 0) AS ratio_drift_{col}_{w}m"
                drift_expressions.append(expr)

    # --- 3. Ensamblar Bloques ---
    final_sql_blocks = [] # Lista para guardar los sub-bloques de SQL

    # Bloque 1: Composición
    if composition_expressions:
        block = "-- Ratios de Composición\n" + ",\n".join(composition_expressions)
        final_sql_blocks.append(block)

    # Bloque 2: Drift
    if drift_expressions:
        block = "-- Ratios de Drift (Actual vs. LAG)\n" + ",\n".join(drift_expressions)
        final_sql_blocks.append(block)
    
    # --- 4. Ensamblar SQL Final ---
    
    if not final_sql_blocks:
        return None
    
    # Unir los bloques de lógica entre sí (ej. Composición Y Drift) con una coma
    body = ",\n\n".join(final_sql_blocks)
    
    return f"\n-- --- 4. Ratios Avanzados (Drift y Composición) ---\n{body}"


def crear_rankings_cero_fijo(cols_to_rank):
    """
    (NUEVO - Reemplaza crear_rankings_percentiles)
    Crea un ranking "cero fijo" (de -1.0 a 1.0) imitando la lógica
    de Pandas de 'full_pipeline.txt'.
    
    - Positivos (val > 0) se rankean de 0.0 a 1.0
    - Negativos (val < 0) se rankean de -1.0 a 0.0
    - Ceros (val = 0) y NULLs se asignan a 0.0
    """
    
    expressions = []
    window_definitions = [] # Esta función ahora también define ventanas

    if not cols_to_rank:
        print("Advertencia: No se generarán rankings. 'cols_to_rank' está vacío.")
        return None, []
    
    for col in cols_to_rank:
        col_casted = f'"{col}"::DOUBLE' # Safe casting
        
        # Definir las ventanas que se usarán
        # w_mes_{col}_pos: Ventana para rankear positivos (ASC)
        # w_mes_{col}_neg: Ventana para rankear negativos (DESC)
        window_name_pos = f"w_mes_{col}_pos"
        window_name_neg = f"w_mes_{col}_neg"
        
        # Definición de la ventana para POSITIVOS
        # Particiona por mes y por el grupo "positivos" (1)
        # Los que no son > 0 (negativos y ceros) quedan en el grupo 0
        window_definitions.append(
            f'{window_name_pos} AS (PARTITION BY foto_mes, CASE WHEN {col_casted} > 0 THEN 1 ELSE 0 END ORDER BY {col_casted} ASC)'
        )
        
        # Definición de la ventana para NEGATIVOS
        # Particiona por mes y por el grupo "negativos" (1)
        window_definitions.append(
            f'{window_name_neg} AS (PARTITION BY foto_mes, CASE WHEN {col_casted} < 0 THEN 1 ELSE 0 END ORDER BY {col_casted} DESC)'
        )

        # Crear la expresión CASE para el SELECT
        # El alias ahora es 'rank_pct_cf_{col}' (cf = cero fijo)
        expression = f"""
        CASE
            WHEN {col_casted} > 0 THEN PERCENT_RANK() OVER {window_name_pos}
            WHEN {col_casted} < 0 THEN -PERCENT_RANK() OVER {window_name_neg}
            ELSE 0.0
        END AS rank_pct_cf_{col}
        """
        expressions.append(expression)

    body = ",\n".join(expressions)
    # Devuelve tanto el SQL para el SELECT como las definiciones para el WINDOW
    return f"\n-- --- 5. Rankings Cero Fijo (Cross-Sectional) ---\n{body}", window_definitions


def get_final_select_list(con, main_table_name, original_cols_list):
    """
    (REFACTORIZADO - Versión robusta)
    Genera la lista SELECT final, aplicando COALESCE a las features nuevas.
    Esto reemplaza la necesidad de 'SELECT * REPLACE'.
    
    MODIFICADO:
    - Añadidas las nuevas features vm_... y vmr_... a la lista de exclusión.
    """
    print("Generando lista SELECT final con imputación...")
    all_cols_df = con.execute(f"DESCRIBE {main_table_name}").fetchdf()
    all_cols = all_cols_df['column_name'].tolist()
    
    select_parts = [] # La lista final de columnas
    
    # Set de columnas "originales" que no deben ser imputadas
    original_cols_set = set(original_cols_list) 
    # Añadir las "combinaciones básicas" que tampoco se imputan
    original_cols_set.update([
        'anio', 'mes', 'mconsumo_total_tarjetas',
        'cproductos_inversion', 'cproductos_seguros',
        'cproductos_prestamos', 
        'ctotal_debitos_automaticos', 'mtotal_debitos_automaticos',
        'ctotal_cheques_rechazados', 'mtotal_cheques_rechazados',
        'peor_estado_tarjetas',
        
        # (NUEVO) Excluir las nuevas combinaciones y ratios de la imputación
        'vm_msaldototal', 'vm_mconsumospesos', 'vm_mlimitecompra',
        'vm_madelantopesos', 'vm_mpagado',
        'vmr_msaldototal_div_mlimitecompra', 'vmr_mconsumospesos_div_mlimitecompra'
    ])
    
    for col in all_cols:
        # Caso 1: Columna original o target. Solo seleccionarla.
        # (Usamos comillas dobles por si los nombres de columnas tienen mayúsculas)
        if col in original_cols_set or col == 'clase_ternaria':
            select_parts.append(f'"{col}"') 
            continue 

        # Caso 2: Es una feature nueva. Aplicar COALESCE.
        default_value = 0 # Imputación por defecto
        
        if col.startswith('ratio_'):
            default_value = 1
        # La nueva 'rank_pct_cf_' no coincide con 'rank_pct_',
        # por lo que caerá correctamente en 'default_value = 0'
        elif col.startswith('rank_pct_'): 
            default_value = 0.5
        
        # Añadir la columna con COALESCE y su alias
        select_parts.append(f'COALESCE("{col}", {default_value}) AS "{col}"')

    # Devolver la lista completa de columnas, unidas por coma
    return ",\n".join(select_parts)


def main():
    """
    (REFACTORIZADO - V3)
    Orquesta el pipeline de Feature Engineering.
    Añade comprobaciones 'if feature_sql:' para evitar errores de comas.
    
    MODIFICADO:
    - Obtiene la lista completa de columnas y la filtra dinámicamente.
    - Pasa la lista de columnas dinámicas a las funciones de FE.
    - Define las ventanas de ranking dinámicamente.
    - Llama a la nueva función de tendencia lineal.
    - Llama a la nueva función de ranking cero fijo y maneja sus ventanas.
    """
    start_time = time.time()
    
    con = duckdb.connect(database=':memory:', read_only=False)
    print("Conexión a DuckDB establecida.")

    try:
        print(f"Cargando datos reparados desde '{REPAIRED_PARQUET_PATH}'...")
        con.execute(f"CREATE VIEW data_reparada AS SELECT * FROM read_parquet('{REPAIRED_PARQUET_PATH}')")
        
        original_cols_df = con.execute("DESCRIBE data_reparada").fetchdf()
        original_cols = original_cols_df['column_name'].tolist()
        
        # --- (NUEVO) Definición de Columnas a Procesar ---
        # Excluir IDs y la variable target, procesar el resto.
        # (Añadir otras columnas de texto si se sabe que existen)
        BLACKLIST_COLS = {'numero_de_cliente', 'foto_mes', 'clase_ternaria'}
        
        # Columnas para deltas, tendencias, ratios de drift y rankings
        # (Básicamente, todas las columnas numéricas o pseudo-numéricas)
        cols_to_process_dynamic = [c for c in original_cols if c not in BLACKLIST_COLS]
        print(f"Se procesarán {len(cols_to_process_dynamic)} columnas dinámicamente para deltas, ratios y rankings.")
        # -------------------------------------------
        
        
        # --- Construcción dinámica de la consulta ---
        select_parts = ["SELECT *"]
        
        window_definitions_cliente = [
            "w_cliente AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)",
            "w_cliente_3m AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)",
            "w_cliente_6m AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)"
        ]
        
        window_definitions_mes = [] 
        se_usaron_ventanas_cliente = False
        se_usaron_ventanas_mes = False

        
        # --- INICIO DEL NUEVO PATRÓN DE ENSAMBLAJE ---
        
        if FE_CONFIG["crear_combinaciones_basicas"]:
            print("Agregando: Combinaciones básicas...")
            feature_sql = crear_combinaciones_basicas()
            if feature_sql:
                select_parts.append(feature_sql)

        if FE_CONFIG["crear_deltas_y_tendencias"]:
            print(f"Agregando: Deltas y Tendencias para ventanas {TIME_WINDOWS}...")
            # MODIFICADO: Pasar la lista de columnas dinámica
            feature_sql = crear_deltas_y_tendencias(cols_to_process_dynamic, TIME_WINDOWS)
            if feature_sql:
                select_parts.append(feature_sql)
                se_usaron_ventanas_cliente = True 

        # (NUEVO) Bloque para tendencias lineales
        if FE_CONFIG["crear_tendencias_lineales"]:
            print(f"Agregando: Tendencias (Regresión Lineal)...")
            feature_sql = crear_tendencias_lineales()
            if feature_sql:
                select_parts.append(feature_sql)
                se_usaron_ventanas_cliente = True 

        if FE_CONFIG["crear_features_categoricas"]:
            print("Agregando: Features categóricas...")
            feature_sql = crear_features_categoricas()
            if feature_sql:
                select_parts.append(feature_sql)
                se_usaron_ventanas_cliente = True 

        if FE_CONFIG["crear_ratios_avanzados"]:
            print(f"Agregando: Ratios Avanzados (Drift) para ventanas {TIME_WINDOWS}...")
            # MODIFICADO: Pasar la lista de columnas dinámica
            feature_sql = crear_ratios_avanzados(cols_to_process_dynamic, TIME_WINDOWS)
            if feature_sql:
                select_parts.append(feature_sql)
                se_usaron_ventanas_cliente = True 

        if FE_CONFIG["crear_rankings_cero_fijo"]: # MODIFICADO: Nombre de la config
            print(f"Agregando: Rankings Cero Fijo (Cross-sectional)...")
            # MODIFICADO: La función ahora devuelve el SQL y las definiciones de ventana
            feature_sql, rank_window_defs = crear_rankings_cero_fijo(cols_to_process_dynamic)
            
            if feature_sql:
                select_parts.append(feature_sql)
                se_usaron_ventanas_mes = True
                # MODIFICADO: Añadir las nuevas definiciones de ventana 
                # (ej. w_mes_mrentabilidad_pos, w_mes_mrentabilidad_neg)
                window_definitions_mes.extend(rank_window_defs)
        
        # --- FIN DEL NUEVO PATRÓN DE ENSAMBLAJE ---
        
        
        # Unir las partes del SELECT.
        # "select_parts" ahora solo contiene "SELECT *" y bloques de features válidos.
        select_clause = ",\n".join(select_parts)
        
        all_windows_parts = []
        if se_usaron_ventanas_cliente:
            all_windows_parts.extend(window_definitions_cliente)
        if se_usaron_ventanas_mes:
            all_windows_parts.extend(window_definitions_mes)

        window_clause = "" 
        if all_windows_parts:
            # Usar set() para evitar duplicados si se definen ventanas en varios lugares
            window_clause = "WINDOW " + ",\n".join(sorted(list(set(all_windows_parts))))
        
        features_query = f"""
        CREATE OR REPLACE TABLE features_calculated_temp AS
        WITH base_data AS (
            SELECT * FROM data_reparada
        )
        {select_clause}
        FROM base_data
        {window_clause};
        """
        
        print("\nCalculando features (Paso 1/2)...")
        # print(f"DEBUG QUERY (Paso 1):\n{features_query}") # Descomentar para debug
        con.execute(features_query)

        # --- LÓGICA DE IMPUTACIÓN (Paso 2/2) ---
        
        final_select_list = get_final_select_list(con, "features_calculated_temp", original_cols)
        
        final_query = f"""
        CREATE TABLE features_dataset AS
        SELECT 
            {final_select_list}
        FROM features_calculated_temp;
        """
        
        print("Imputando nulos y creando tabla final (Paso 2/2)...")
        # print(f"DEBUG QUERY (Paso 2):\n{final_query}") # Descomentar para debug
        con.execute(final_query)
        
        print("Tabla 'features_dataset' creada en memoria.")

        # --- Guardar los Datos con Features ---
        print(f"Guardando dataset de features en '{FEATURES_PARQUET_PATH}'...")
        con.execute(f"COPY (SELECT * FROM features_dataset) TO '{FEATURES_PARQUET_PATH}' (FORMAT 'PARQUET', CODEC 'ZSTD')")
        
        end_time = time.time()
        print(f"\n--- ¡Proceso de Feature Engineering completado en {end_time - start_time:.2f} segundos! ---")
        print(f"Archivo final listo para el modelo: {FEATURES_PARQUET_PATH}")
        
        # Verificar el esquema final
        print("\nEsquema del archivo de features (primeras 20 columnas):")
        con.execute(f"DESCRIBE SELECT * FROM '{FEATURES_PARQUET_PATH}' LIMIT 1")
        print(con.fetchdf().head(20))
        print("...")

    except Exception as e:
        print(f"\nHa ocurrido un error durante el proceso de feature engineering: {e}")

    finally:
        con.close()
        print("Conexión a DuckDB cerrada.")


if __name__ == "__main__":
    main()
