# ==============================/ preprocessing.py /=====================================
#   - Feito por: Manuele Christófalo
#   - Aplicado na pesquisa: "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE
#      SISTEMAS EMBARCADOS PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON"
#
#   --> Normalização dos dados e criação de janelas de análise
# =======================================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode as scipy_mode

# 1. Normalização através do Standard Scaler (Padronizador) -------------------
def get_scaler(df_train: pd.DataFrame, features: list) -> StandardScaler:
    """
    Cria e 'fita' um StandardScaler APENAS nos dados de treino.
    """
    scaler = StandardScaler()
    scaler.fit(df_train[features])
    return scaler

def scale_data(df: pd.DataFrame, scaler: StandardScaler, features: list) -> pd.DataFrame:
    """Aplica um scaler já 'fitado' aos dados."""
    df_scaled = df.copy()
    df_scaled[features] = scaler.transform(df_scaled[features])
    return df_scaled


# 2. Criação das janelas (5s cada) --------------------------------------------
def create_sequences(df: pd.DataFrame, 
                       features: list, 
                       target_col: str, 
                       window_size: int, 
                       step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cria janelas deslizantes de dados (X) e seus respectivos rótulos (y).
    Itera sobre cada coleta para não misturar dados.
    """
    all_X, all_y = [], []
    
    for coleta_id in df['ID_Coleta'].unique():
        df_coleta = df[df['ID_Coleta'] == coleta_id]
        
        data_values = df_coleta[features].values
        labels = df_coleta[target_col].values
        
        for i in range(0, len(data_values) - window_size, step):
            # X = A janela de dados (ex: 50 amostras, 4 features)
            window = data_values[i : i + window_size]
            
            # y = O rótulo para essa janela
            # Usamos a "moda" (valor mais frequente) da flag 'Tremor'
            # dentro da janela. Se 60% da janela for '1', o rótulo é '1'.
            window_labels = labels[i : i + window_size]
            label = scipy_mode(window_labels, keepdims=False)[0]
            
            all_X.append(window)
            all_y.append(label)
            
    return np.array(all_X), np.array(all_y)