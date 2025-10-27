# =================================/ model.py /==========================================
#   - Feito por: Manuele Christófalo
#   - Aplicado na pesquisa: "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE
#      SISTEMAS EMBARCADOS PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON"
#
#   --> Define (modela) a arquitetura LSTM
# =======================================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# 1. Criação do modelo --------------------------------------------------------
def build_model(window_size: int, n_features: int) -> Sequential:
    """
    Constrói a arquitetura do modelo LSTM para classificação binária.
    """
    model = Sequential() # Tipo sequencial
    
    model.add(Input(shape=(window_size, n_features)))
    
    # Camada LSTM (pode-se adicionar mais)
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(32, return_sequences=False)) # Última LSTM não retorna sequência
    model.add(Dropout(0.2))
    
    # Camada Densa para interpretação
    model.add(Dense(16, activation='relu'))
    
    # Camada de saída para classificação binária
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# 2. Compila o modelo usando Adam ---------------------------------------------
def compile_model(model: Sequential, learning_rate: float = 0.001):
    """
    Compila o modelo com otimizador, perda e métricas.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy', # Correto para classificação binária
        metrics=['accuracy']
    )
    return model