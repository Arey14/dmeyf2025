import duckdb
import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np 
from sklearn.model_selection import train_test_split

# --- 1. Definición de Ganancia ---
PROFIT_TP = 780000  # Ganancia por Verdadero Positivo (detectar un BAJA+2)
COST_FP = -20000    # Costo por Falso Positivo (molestar a un cliente CONTINUA)
# ---------------------------------

# --- 2. Métrica de Ganancia Personalizada ---
def lgbm_profit_metric(y_true, y_pred_probs):
    """
    Métrica de ganancia personalizada para LightGBM.
    Prueba 50 umbrales (thresholds) y devuelve la ganancia máxima.
    """
    # Convertir a arrays de numpy
    y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    
    max_profit = -np.inf  # Empezamos con la peor ganancia posible
    
    # Probar umbrales de 0.01 a 0.50
    thresholds = np.linspace(0.01, 0.50, 50)
    
    for thresh in thresholds:
        # Convertir probabilidades en predicciones (0 o 1)
        y_pred = (y_pred_probs > thresh).astype(int)
        
        # Calcular Verdaderos Positivos (TP) y Falsos Positivos (FP)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Calcular la ganancia total para este umbral
        total_profit = (tp * PROFIT_TP) + (fp * COST_FP)
        
        if total_profit > max_profit:
            max_profit = total_profit
            
    # LightGBM maximiza esta métrica
    # (Nombre de la métrica, valor, ¿es más alto mejor?)
    return 'max_profit', max_profit, True
# ---------------------------------


# --- 3. Configuración de Archivos ---
FEATURES_PARQUET_PATH = 'features_dataset.parquet'
IMPORTANCE_OUTPUT_PATH = 'feature_importances_avg.csv'

# --- 4. Configuración del Split (Time-Series) ---
# ( (train_start, train_end), validation_month )
CV_FOLDS = [
    ( (202007, 202101), 202102 ), # Fold 1: Entrena en 7 meses, Valida en 202102
    ( (202007, 202103), 202104 )  # Fold 2: Entrena en 9 meses, Valida en 202104
]
GLOBAL_EXCLUDE_MONTHS = [ 202006 ] # Mes anómalo

# --- ¡NUEVO! Configuración de Semillas ---
N_SEEDS = [761249, 762001, 763447, 762233, 761807,800003, 800021, 800051, 800093, 800123,100057, 100207, 100237, 100267, 100297,254881, 499987, 749987, 900007, 378553]
# -----------------------------------------

print(f"--- Iniciando Selección de Features (CV Folds={len(CV_FOLDS)}, Seeds={len(N_SEEDS)}) ---")
print(f"Dataset de entrada: {FEATURES_PARQUET_PATH}")

if not os.path.exists(FEATURES_PARQUET_PATH):
    print(f"Error: No se encontró el archivo de features '{FEATURES_PARQUET_PATH}'.")
    print("Asegúrate de que 'feature_engineering.py' se haya ejecutado exitosamente.")
    sys.exit(1)

# Conectar a DuckDB
con = duckdb.connect(database=':memory:', read_only=False)
print("Conexión a DuckDB establecida.")

# --- 5. Preparación de Datos (Funciones Helper) ---
def fix_dtypes_and_target(df):
    """Aplica dtypes, casting y mapeo de target."""
    # print("Forzando conversión de tipos (casting)...") # Comentado para reducir logs
    if 'cmobile_app_trx_pre202010' in df.columns:
        df['cmobile_app_trx_pre202010'] = pd.to_numeric(df['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df.columns:
        df['cmobile_app_trx_post202010'] = pd.to_numeric(df['cmobile_app_trx_post202010'], errors='coerce').fillna(0)

    # Mapeo del Target (BINARIO)
    target = 'clase_ternaria'
    df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
    df[target] = df[target].fillna(0).astype(int)
    
    return df, target

def get_features_and_cats(df):
    """Obtiene la lista de features y columnas categóricas."""
    cols_to_drop = [
        'numero_de_cliente', 'foto_mes', 'clase_ternaria',
        'Master_status', 'Visa_status' 
    ]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    features = [col for col in df.columns if col not in cols_to_drop]
    
    cat_features = [
        'anio', 'mes', 'peor_estado_tarjetas', 'active_quarter', 'cliente_vip',
        'internet', 'tcallcenter', 'thomebanking', 'tmobile_app', 
        'Master_delinquency', 'Visa_delinquency', 'ccaja_seguridad'
    ]
    
    cat_features_final = []
    for col in cat_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            df[col] = df[col].astype('category')
            cat_features_final.append(col)
            
    return features, cat_features_final
# -------------------------------------------------

try:
    # --- 6. Bucle de Entrenamiento (¡MODIFICADO PARA EFICIENCIA DE RAM!) ---
    
    all_importances = [] # Lista para guardar los DFs de importancia (total: 2 * 20 = 40)

    print(f"\nIniciando entrenamiento en bucle anidado...")

    # Obtener la lista de features y categóricas UNA SOLA VEZ
    print("Cargando esquema de features (leyendo 1 fila)...")
    df_sample = con.execute(f"SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}') LIMIT 1").fetchdf()
    df_sample, target = fix_dtypes_and_target(df_sample)
    features, cat_features_final = get_features_and_cats(df_sample)
    print(f"Total de features a entrenar: {len(features)}")
    del df_sample # Liberar memoria

    # --- INICIO DEL BUCLE EXTERNO (CV FOLDS) ---
    for i, (train_window, val_month) in enumerate(CV_FOLDS):
        fold_num = i + 1
        train_start_month, train_end_month = train_window
        
        print(f"\n===========================================================")
        print(f"--- Cargando Datos para Fold {fold_num}/{len(CV_FOLDS)} ---")
        print(f"  Entrenamiento: foto_mes >= {train_start_month} Y <= {train_end_month}")
        print(f"  Validación:    foto_mes == {val_month}")
        
        # --- Carga de datos "Just-in-Time" ---
        months_to_load = [val_month] + list(range(train_start_month, train_end_month + 1))
        months_to_exclude = GLOBAL_EXCLUDE_MONTHS
        
        sql_load_fold = f"""
        SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')
        WHERE foto_mes IN ({', '.join(map(str, months_to_load))})
        AND foto_mes NOT IN ({', '.join(map(str, months_to_exclude))})
        """
        
        print(f"Cargando datos para Fold {fold_num} en memoria...")
        df_fold = con.execute(sql_load_fold).fetchdf()
        
        # Aplicar preparación de datos SÓLO a este subset
        df_fold, target = fix_dtypes_and_target(df_fold)
        for col in cat_features_final:
             df_fold[col] = pd.to_numeric(df_fold[col], errors='coerce').fillna(0).astype(int)
             df_fold[col] = df_fold[col].astype('category')
        
        print(f"Datos del fold preparados. Forma: {df_fold.shape}")
        
        # --- Crear Splits para este Fold ---
        train_idx = (df_fold['foto_mes'] >= train_start_month) & (df_fold['foto_mes'] <= train_end_month)
        X_train = df_fold.loc[train_idx, features]
        y_train = df_fold.loc[train_idx, target]

        val_idx = df_fold['foto_mes'] == val_month
        X_val = df_fold.loc[val_idx, features]
        y_val = df_fold.loc[val_idx, target]

        print(f"  Tamaño de Entrenamiento: {X_train.shape[0]} registros")
        print(f"  Tamaña de Validación:   {X_val.shape[0]} registros")
        
        # Liberar el DataFrame grande del fold, nos quedamos solo con train/val
        del df_fold 
        
        if X_val.empty or X_train.empty:
            print(f"¡ADVERTENCIA! El set de entrenamiento o validación para el Fold {fold_num} está vacío. Saltando este fold.")
            continue
        
        # --- INICIO DEL BUCLE INTERNO (SEEDS) ---
        print(f"Iniciando bucle de {len(N_SEEDS)} seeds para el Fold {fold_num}...")
        for j, seed in enumerate(N_SEEDS):
            seed_num = j + 1
            print(f"  --- Entrenando Fold {fold_num}, Seed {seed_num}/{len(N_SEEDS)} (Seed={seed}) ---")

            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                metric='none', 
                n_estimators=1000,
                learning_rate=0.05,
                n_jobs=-1,
                seed=seed, # <-- ¡Semilla variable!
                colsample_bytree=0.7,
                subsample=0.7,
                reg_alpha=0.1,
                reg_lambda=0.1,
                is_unbalanced=True
            )

            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=lgbm_profit_metric, # <-- Métrica de Ganancia
                callbacks=[lgb.early_stopping(100, verbose=False)],
                categorical_feature=cat_features_final
            )
            
            best_profit = lgb_model.best_score_['valid_0']['max_profit']
            print(f"    Mejor Ganancia (Fold {fold_num}, Seed {seed}): ${best_profit:,.0f}")

            # --- Extraer Importancias (por Seed) ---
            df_importance_fold = pd.DataFrame({
                'feature': features,
                'importance_gain': lgb_model.feature_importances_
            })
            all_importances.append(df_importance_fold)

    print("\nEntrenamiento de todos los Folds y Seeds completado.")

    # --- 9. Agregar Resultados ---
    print("Agregando importancias de todos los folds...")
    
    if not all_importances:
        print("¡Error! No se pudo entrenar ningún fold. No se generará el archivo de importancia.")
        sys.exit(1)
        
    df_all_importances = pd.concat(all_importances)
    
    # Calcular media, std dev, y cuántas veces fue "usada"
    df_agg = df_all_importances.groupby('feature')['importance_gain'].agg(
        mean_importance_gain='mean',
        std_importance_gain='std',
        count_used=lambda x: (x > 0).sum()
    ).reset_index()
    
    df_agg = df_agg.sort_values(by='mean_importance_gain', ascending=False).reset_index(drop=True)
    
    # Guardar en CSV
    df_agg.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)
    
    print(f"\n--- ¡Selección de Features Completada! ---")
    print(f"Lista agregada (promedio, std) guardada en: {IMPORTANCE_OUTPUT_PATH}")
    
    print(f"\n--- TOP 50 FEATURES MÁS INFLUYENTES (Promedio de {len(all_importances)} runs) ---")
    print(df_agg.head(50).to_string())
    print("---------------------------------------------------------------")


except Exception as e:
    print(f"\nHa ocurrido un error durante el proceso de selección: {e}")

finally:
    if 'con' in locals() or 'con' in globals():
        con.close()
        print("\nConexión a DuckDB cerrada.")
