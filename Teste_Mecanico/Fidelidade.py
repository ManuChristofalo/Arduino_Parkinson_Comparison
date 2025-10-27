# ==================================/ Análise de Fidelidade /=========================================                                                     
#     - Autora: Manuele S. Christofalo
#     - Data: 30/09/2025
#     - Feito para o TCC "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE SISTEMAS EMBARCADOS
#      PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON "
#
#     -> Após os experimentos, esse código analisa os dados obtidos em termos de fidelidade
#       (frequencia fixa) para encontrar possíveis ruídos ou lags atrelados aos sensores.
#  ===================================================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- PARÂMETROS DO TESTE ---
ARQUIVO_UNO = 'data/pUNO_TesteB_10g.csv'
ARQUIVO_NANO = 'data/pNANO_TesteB_10g.csv'
GROUND_TRUTH_FREQ = 4.0   #4.0 Hz (fixo)
GROUND_TRUTH_AMP = 10.0   #10, 20 ou 30 graus
EIXO_MOVIMENTO = 'pitch'  #Eixo em que o servo moveu


# 1. Carrega e prepara os dados -----------------------------------------------
def carregar_e_preparar(arquivo):
    """Carrega o CSV, renomeia colunas e extrai o sinal e o tempo."""
    df = pd.read_csv(arquivo, header=0)
    
    # Renomeia as colunas
    df.rename(columns={
        'Roll (x)': 'roll',
        ' Pitch (Y)': 'pitch',
        ' Yaw (Z)': 'yaw',
        ' Time (s)': 'tempo_s'
    }, inplace=True)
    
    sinal = df[EIXO_MOVIMENTO].values
    tempo = df['tempo_s'].values
    return sinal, tempo


# 2. Execução Principal -------------------------------------------------------
print("--- Script 2: Análise de Fidelidade (Domínio do Tempo) ---")

sinal_uno, tempo_uno = carregar_e_preparar(ARQUIVO_UNO)
sinal_nano, tempo_nano = carregar_e_preparar(ARQUIVO_NANO)

tempo_ref = tempo_nano #Escolha arbitrária

# A. Gerar o Sinal Ground Truth Perfeito ----
sinal_truth = GROUND_TRUTH_AMP * np.sin(2 * np.pi * GROUND_TRUTH_FREQ * tempo_ref)

# B. Normalizar os Sinais (StandardScaler) ----
# Isso é crucial para comparar a FORMA da onda, não a amplitude absoluta.
scaler = StandardScaler()

# Reshape (-1, 1) é necessário para o scaler
sinal_truth_scaled = scaler.fit_transform(sinal_truth.reshape(-1, 1))
sinal_uno_scaled = scaler.transform(sinal_uno.reshape(-1, 1))
sinal_nano_scaled = scaler.transform(sinal_nano.reshape(-1, 1))

# C. Calcular RMSE (Erro Médio Quadrático) ----
rmse_uno = np.sqrt(mean_squared_error(sinal_truth_scaled, sinal_uno_scaled))
rmse_nano = np.sqrt(mean_squared_error(sinal_truth_scaled, sinal_nano_scaled))


# --- Saída Quantitativa (Tabela no Console) ---
print("\n--- Tabela de Resultados (Fidelidade do Sinal) ---")
print("=" * 40)
print(f"| Métrica                 | pUNO_v2 | pNANO_v2 |")
print("-" * 40)
print(f"| RMSE vs Ground Truth  | {rmse_uno:<7.4f} | {rmse_nano:<8.4f} |")
print("=" * 40)
print("(Valores menores de RMSE indicam maior fidelidade à forma da onda)")

# --- Saída Gráfica (Visualização) ---
plt.figure(figsize=(15, 7))
plt.title(f'Comparação de Fidelidade de Sinal (Normalizado) - {GROUND_TRUTH_FREQ} Hz')
plt.plot(tempo_ref, sinal_truth_scaled, 'k--', label='Ground Truth (Senoide Perfeita)', linewidth=2)
plt.plot(tempo_ref, sinal_uno_scaled, 'r-', label=f'pUNO (RMSE: {rmse_uno:.4f})', alpha=0.7)
plt.plot(tempo_ref, sinal_nano_scaled, 'b-', label=f'pNANO (RMSE: {rmse_nano:.4f})', alpha=0.7)

# Limita o plot para ver o detalhe (ex: 2 segundos)
plt.xlim(5, 7) 
plt.xlabel('Tempo (s)')
plt.ylabel('Sinal Normalizado (Z-score)')
plt.legend()
plt.grid(True)
plt.show()