# ================================/ plotting.py /========================================
#   - Feito por: Manuele Christófalo
#   - Aplicado na pesquisa: "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE
#      SISTEMAS EMBARCADOS PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON"
#
#   --> Funções gráficas para normalização, treinamento e aplicação da lstm
# =======================================================================================


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Gráfico da normalização --------------------------------------------------
def plot_normalized_data(df_coleta: pd.DataFrame, features: list, n_samples: int = 1000):
    """
    (Etapa 1) Plota os primeiros 'n_samples' dos dados normalizados de uma coleta.
    """
    plt.figure(figsize=(18, 8))
    
    plot_data = df_coleta.head(n_samples)
    
    for feature in features:
        plt.plot(plot_data['Time (s)'], plot_data[feature], label=feature, alpha=0.8)
        
    plt.title(f'Dados Normalizados (Primeiras {n_samples // 10} segundos da Coleta {df_coleta["ID_Coleta"].iloc[0]})', fontsize=16)
    plt.xlabel('Tempo (s)', fontsize=12)
    plt.ylabel('Valor Normalizado (Z-score)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# 2. Gráfico do treinamento ---------------------------------------------------
def plot_training_history(history):
    """
    (Etapa 2) Plota as curvas de perda (Loss) e acurácia (Accuracy) do treino.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Gráfico de Perda (Loss)
    ax1.plot(history.history['loss'], label='Perda (Treino)')
    ax1.plot(history.history['val_loss'], label='Perda (Validação)')
    ax1.set_ylabel('Perda (Loss)', fontsize=12)
    ax1.set_title('Histórico de Treinamento - Perda', fontsize=16)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfico de Acurácia (Accuracy)
    ax2.plot(history.history['accuracy'], label='Acurácia (Treino)')
    ax2.plot(history.history['val_accuracy'], label='Acurácia (Validação)')
    ax2.set_xlabel('Épocas', fontsize=12)
    ax2.set_ylabel('Acurácia', fontsize=12)
    ax2.set_title('Histórico de Treinamento - Acurácia', fontsize=16)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


# 3. Gráfico da predição ------------------------------------------------------
def plot_predictions(y_true: np.ndarray, y_pred_classes: np.ndarray, title: str, n_samples: int = 1500):
    """
    (Etapa 3) Compara os rótulos verdadeiros com as previsões do modelo.
    """
    plt.figure(figsize=(18, 8))
    
    # Pega apenas uma fatia para visualização
    y_true_slice = y_true[:n_samples]
    y_pred_slice = y_pred_classes[:n_samples]
    
    time_axis = np.arange(len(y_true_slice))
    
    # Plota a verdade (Ground Truth)
    plt.plot(time_axis, y_true_slice, label='Verdadeiro (Tremor Real)', 
             color='blue', linewidth=2, drawstyle='steps-post')
    
    # Plota a previsão
    # Usamos +0.05 de offset para ver a sobreposição
    plt.plot(time_axis, y_pred_slice + 0.05, label='Previsão (Tremor Previsto)', 
             color='orange', linewidth=2, alpha=0.8, drawstyle='steps-post')
    
    plt.title(title, fontsize=16)
    plt.xlabel(f'Índice da Janela (Primeiras {n_samples} janelas)', fontsize=12)
    plt.ylabel('Classe (0 = Normal, 1 = Tremor)', fontsize=12)
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()