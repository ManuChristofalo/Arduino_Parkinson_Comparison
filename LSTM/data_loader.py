# ===============================/ data_loader.py /======================================
#   - Feito por: Manuele Christófalo
#   - Aplicado na pesquisa: "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE
#      SISTEMAS EMBARCADOS PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON"
#
#   --> Carrega o CSV, unifica as métricas e divide os dados para treino e teste
# =======================================================================================

import pandas as pd
import numpy as np

# 1. Carrega os dados ---------------------------------------------------------
def load_data(csv_path: str) -> pd.DataFrame:
    """Carrega os dados do arquivo CSV."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {csv_path}")
        return pd.DataFrame()
    
    # Limpa nomes de colunas, se necessário
    df.columns = [col.strip() for col in df.columns]
    return df


# 2. Unifica as leituras em uma só (Magnitude) --------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a feature 'Magnitude' unificando as métricas
    """
    df_copy = df.copy()
    df_copy['Magnitude'] = np.sqrt(
        df_copy['Roll (x)']**2 + 
        df_copy['Pitch (y)']**2 + 
        df_copy['Yaw (z)']**2
    )
    return df_copy


# 3. Divide os dados (coletas) em conjunto de treino e teste ------------------
def split_data_by_coleta(df: pd.DataFrame, train_ids: list, test_ids: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o DataFrame principal em treino e teste com base nos IDs de Coleta."""
    df_train = df[df['ID_Coleta'].isin(train_ids)].copy()
    df_test = df[df['ID_Coleta'].isin(test_ids)].copy()
    
    print(f"Coletas de Treino: {train_ids} (Total de {len(df_train)} linhas)")
    print(f"Coletas de Teste: {test_ids} (Total de {len(df_test)} linhas)")
    
    return df_train, df_test