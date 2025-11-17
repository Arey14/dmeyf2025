import duckdb
import os
import sys
import pandas as pd
import lightgbm as lgb
import numpy as np 
from sklearn.model_selection import train_test_split
import json # Para cargar los hiperparámetros

# --- 1. Definición de Ganancia ---
PROFIT_TP = 780000  # Ganancia por Verdadero Positivo (detectar un BAJA+2)
COST_FP = -20000    # Costo por Falso Positivo (molestar a un cliente CONTINUA)
# ---------------------------------

# --- 2. Métrica de Ganancia Personalizada (para LGBM) ---
def lgbm_profit_metric(y_true, y_pred_probs):
    """
    Métrica de ganancia personalizada para LightGBM.
    Encuentra el umbral que maximiza la ganancia.
    """
    y_true = y_true.get_label() if hasattr(y_true, 'get_label') else y_true
    
    max_profit = -np.inf
    best_thresh = 0.5 # Default
    
    thresholds = np.linspace(0.01, 0.50, 50)
    
    for thresh in thresholds:
        y_pred = (y_pred_probs > thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_profit = (tp * PROFIT_TP) + (fp * COST_FP)
        
        if total_profit > max_profit:
            max_profit = total_profit
            best_thresh = thresh
            
    # Guardamos el mejor umbral en el modelo para usarlo después
    lgbm_profit_metric.best_thresh = best_thresh
    
    return 'max_profit', max_profit, True
# ---------------------------------

# --- 3. Configuración de Archivos y Parámetros ---
FEATURES_PARQUET_PATH = 'features_dataset.parquet'

# ¡IMPORTANTE! Usamos los resultados de nuestros scripts anteriores
IMPORTANCE_INPUT_PATH = 'feature_importances_conjunto.json' # De seleccion_features.py
HYPERPARAMS_INPUT_PATH = 'best_hyperparams.json'        # De optimizar_hiperparametros.py
TOP_N_FEATURES = 1300 # Usar las 100 mejores features

# Meses corruptos a excluir SIEMPRE
GLOBAL_EXCLUDE_MONTHS = [ 202006 ] # Mes anómalo
# ---------------------------------

def fix_dtypes(df):
    """Aplica las correcciones de dtype que encontramos."""
    if 'cmobile_app_trx_pre202010' in df.columns:
        df['cmobile_app_trx_pre202010'] = pd.to_numeric(df['cmobile_app_trx_pre202010'], errors='coerce').fillna(0)
    if 'cmobile_app_trx_post202010' in df.columns:
        df['cmobile_app_trx_post202010'] = pd.to_numeric(df['cmobile_app_trx_post202010'], errors='coerce').fillna(0)
    
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
    return df, cat_features_final

def prepare_data(df, top_features_list):
    """Prepara el DataFrame completo para el modelado."""
    
    print("Preparando datos (Dtypes, Target Binario)...")
    df, cat_features_final = fix_dtypes(df)

    # Codificar el Target (BINARIO)
    target = 'clase_ternaria'
    df[target] = df[target].map({'BAJA+2': 1, 'BAJA+1': 0, 'CONTINUA': 0})
    df[target] = df[target].fillna(0).astype(int)
    
    # Filtrar solo las features seleccionadas
    features = [f for f in top_features_list if f in df.columns]
    cat_features_final = [f for f in cat_features_final if f in features]
    
    print(f"Usando {len(features)} features (Top {TOP_N_FEATURES}) y {len(cat_features_final)} features categóricas.")
    
    return df, features, cat_features_final, target

def run_prediction_pipeline(df, features, cat_features, target, base_hyperparams, train_end_month, val_month, pred_month, output_filename):
    """
    Ejecuta un ciclo completo de entrenamiento y predicción.
    1. Entrena en [..., train_end_month] y valida en [val_month] para encontrar el umbral.
    2. Re-entrena en [..., val_month].
    3. Predice en [pred_month] y guarda.
    """
    print(f"\n--- Iniciando Predicción para {pred_month} ---")
    
    # --- 1. Búsqueda de Hiperparámetros (Umbral e Iteraciones) ---
    print(f"Buscando hiperparámetros: Entrenando hasta {train_end_month}, Validando en {val_month}...")
    
    train_idx = (df['foto_mes'] <= train_end_month) & (~df['foto_mes'].isin(GLOBAL_EXCLUDE_MONTHS))
    val_idx = df['foto_mes'] == val_month
    
    X_train_hyper = df.loc[train_idx, features]
    y_train_hyper = df.loc[train_idx, target]
    X_val_hyper = df.loc[val_idx, features]
    y_val_hyper = df.loc[val_idx, target]
    
    # Usar los hiperparámetros base encontrados por Optuna
    lgb_model_hyper = lgb.LGBMClassifier(
        **base_hyperparams,
        n_estimators=1000, # Usar un n_estimators alto para el early stopping
        n_jobs=-1,
        seed=42,
        is_unbalanced=True
    )
    
    lgb_model_hyper.fit(
        X_train_hyper, y_train_hyper,
        eval_set=[(X_val_hyper, y_val_hyper)],
        eval_metric=lgbm_profit_metric,
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=cat_features
    )
    
    best_iteration = lgb_model_hyper.best_iteration_
    best_threshold = lgbm_profit_metric.best_thresh
    best_profit = lgb_model_hyper.best_score_['valid_0']['max_profit']
    
    print(f"  Mejor Ganancia encontrada: ${best_profit:,.0f}")
    print(f"  Mejor Umbral (Threshold): {best_threshold:.4f}")
    print(f"  Mejores Iteraciones: {best_iteration}")

    # --- 2. Entrenamiento del Modelo Final ---
    print(f"Entrenando modelo final: Datos hasta {val_month}...")
    
    # Incluir el set de validación en el entrenamiento final
    final_train_idx = (df['foto_mes'] <= val_month) & (~df['foto_mes'].isin(GLOBAL_EXCLUDE_MONTHS))
    X_train_final = df.loc[final_train_idx, features]
    y_train_final = df.loc[final_train_idx, target]
    
    lgb_model_final = lgb.LGBMClassifier(
        **base_hyperparams,
        n_estimators=best_iteration, # Usar la mejor iteración
        n_jobs=-1,
        seed=42,
        is_unbalanced=True
    )
    
    lgb_model_final.fit(
        X_train_final, y_train_final,
        categorical_feature=cat_features
    )
    
    print("Modelo final entrenado.")

    # --- 3. Generación de Predicciones ---
    print(f"Generando predicciones para {pred_month}...")
    
    pred_idx = df['foto_mes'] == pred_month
    X_pred = df.loc[pred_idx, features]
    
    if X_pred.empty:
        print(f"¡Error! No se encontraron datos para el mes de predicción {pred_month}.")
        return

    # Predecir probabilidades
    pred_probs = lgb_model_final.predict_proba(X_pred)[:, 1]
    
    # Aplicar el umbral de ganancia
    pred_labels = (pred_probs > best_threshold).astype(int)
    
    # Crear el DataFrame de entrega
    df_entrega = pd.DataFrame({
        'numero_de_cliente': df.loc[pred_idx, 'numero_de_cliente'],
        'foto_mes': pred_month,
        'probabilidad_baja': pred_probs,
        'prediccion_baja': pred_labels
    })
    
    df_entrega.to_csv(output_filename, index=False)
    print(f"¡Predicciones guardadas en '{output_filename}'!")
    print(f"  Total de clientes a enviar (predicción=1): {df_entrega['prediccion_baja'].sum()} de {len(df_entrega)}")
    

def main():
    print(f"--- Iniciando Pipeline de Predicción Final ---")
    
    # --- Cargar Datos y Features ---
    if not os.path.exists(FEATURES_PARQUET_PATH) or not os.path.exists(IMPORTANCE_INPUT_PATH) or not os.path.exists(HYPERPARAMS_INPUT_PATH):
        print(f"Error: Faltan archivos. Asegúrate de que '{FEATURES_PARQUET_PATH}', '{IMPORTANCE_INPUT_PATH}' y '{HYPERPARAMS_INPUT_PATH}' existan.")
        sys.exit(1)

    con = duckdb.connect(database=':memory:', read_only=False)
    print("Conexión a DuckDB establecida.")
    
    try:
        # Cargar todos los datos
        df = con.execute(f"SELECT * FROM read_parquet('{FEATURES_PARQUET_PATH}')").fetchdf()
        
        # Cargar la lista de Top N features
        df_top_features = pd.read_json(IMPORTANCE_INPUT_PATH) # Asumiendo JSON del último script
        top_features_list = df_top_features.head(TOP_N_FEATURES)['feature'].tolist()
        
        # Cargar los mejores hiperparámetros
        with open(HYPERPARAMS_INPUT_PATH, 'r') as f:
            base_hyperparams = json.load(f)
        print(f"Hiperparámetros base cargados desde '{HYPERPARAMS_INPUT_PATH}':")
        print(base_hyperparams)

        # Preparar el DataFrame (Dtypes, Target, etc.)
        df, features, cat_features_final, target = prepare_data(df, top_features_list)

        # --- RUN 1: PREDECIR ABRIL 2021 ---
        # (Entrena hasta 202101, Valida en 202102)
        run_prediction_pipeline(
            df=df, features=features, cat_features=cat_features_final, target=target,
            base_hyperparams=base_hyperparams,
            train_end_month=202101,
            val_month=202102,
            pred_month=202104,
            output_filename="predicciones_abril.csv"
        )

        # --- RUN 2: PREDECIR JUNIO 2021 ---
        # (Entrena hasta 202103, Valida en 202104)
        run_prediction_pipeline(
            df=df, features=features, cat_features=cat_features_final, target=target,
            base_hyperparams=base_hyperparams,
            train_end_month=202103,
            val_month=202104,
            pred_month=202106,
            output_filename="predicciones_junio.csv"
        )
        
        print("\n--- Pipeline de Predicción Completado ---")

    except Exception as e:
        print(f"\nHa ocurrido un error durante el proceso de predicción: {e}")

    finally:
        if 'con' in locals() or 'con' in globals():
            con.close()
            print("\nConexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()
