# ==================================/ Análise de Frequência /=========================================                                                     
#     - Autora: Manuele S. Christofalo
#     - Data: 30/09/2025
#     - Feito para o TCC "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE SISTEMAS EMBARCADOS
#      PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON "
#
#     -> Após os experimentos, esse código analisa os dados obtidos em termos de frequência
#       (amplitude fixa) através da Transformada Rápida de Fourrier.
#  ===================================================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- PARÂMETROS DO TESTE ---
# Altere estes valores para cada ensaio
ARQUIVO_UNO = 'data/pUNO_TesteA_2Hz.csv'
ARQUIVO_NANO = 'data/pNANO_TesteA_2Hz.csv'
GROUND_TRUTH_FREQ = 2.0  # Frequência (em Hz) que o servo motor gerou (2, 3, 4 ou 4.5*)


# 1. Carrega e prepara os dados -----------------------------------------------
def carregar_dados(arquivo):
    """Carrega o CSV, renomeia colunas e calcula a taxa de amostragem (fs)."""
    df = pd.read_csv(arquivo, header=0) 
    
    # Renomeia as colunas
    df.rename(columns={
        'Roll (x)': 'roll',
        ' Pitch (Y)': 'pitch',
        ' Yaw (Z)': 'yaw',
        ' Time (s)': 'tempo_s'
    }, inplace=True)

    # Calcula a Magnitude
    df['magnitude'] = np.sqrt(df['roll']**2 + df['pitch']**2 + df['yaw']**2)
    
    # Calcula a taxa de amostragem (fs)
    delta_t = df['tempo_s'].diff().mean()
    fs = 1.0 / delta_t
    
    return df, fs


# 2. FFT (Fast Fourrier Transform): Tempo -> Frequencia -----------------------
def analisar_fft(df, fs):
    """Executa a FFT no sinal de magnitude e encontra o pico de frequência."""
    sinal = df['magnitude'].values
    N = len(sinal)
    
    # Normaliza o sinal removendo a média (componente DC)
    sinal = sinal - np.mean(sinal)
    
    # Calcula a FFT
    yf = fft(sinal)
    xf = fftfreq(N, 1 / fs) # Gera os "bins" de frequência
    
    # Pega apenas a metade positiva do espectro
    xf_positive = xf[:N//2]
    yf_positive = 2.0/N * np.abs(yf[0:N//2])
    
    # Encontra o índice do pico de magnitude (ignorando o primeiro bin, que é 0 Hz)
    peak_index = np.argmax(yf_positive[1:]) + 1 
    freq_medida = xf_positive[peak_index]
    
    return xf_positive, yf_positive, freq_medida


# 3. Execução Principal -------------------------------------------------------
print("--- Script 1: Análise de Frequência (FFT) ---")

# Carrega e processa dados
df_uno, fs_uno = carregar_dados(ARQUIVO_UNO)
df_nano, fs_nano = carregar_dados(ARQUIVO_NANO)

# Analisa FFT
xf_uno, yf_uno, freq_uno = analisar_fft(df_uno, fs_uno)
xf_nano, yf_nano, freq_nano = analisar_fft(df_nano, fs_nano)

# Calcula erros
erro_uno = ((freq_uno - GROUND_TRUTH_FREQ) / GROUND_TRUTH_FREQ) * 100
erro_nano = ((freq_nano - GROUND_TRUTH_FREQ) / GROUND_TRUTH_FREQ) * 100


# --- Saída Quantitativa (Tabela no Console) ---
print("\n--- Tabela de Resultados (Frequência) ---")
print("=" * 60)
print(f"| Ground Truth | Sistema | Freq. Medida | Erro Relativo (%) |")
print("-" * 60)
print(f"| {GROUND_TRUTH_FREQ:<12.2f} | pUNO_v2 | {freq_uno:<12.2f} | {erro_uno:<17.2f} |")
print(f"| {GROUND_TRUTH_FREQ:<12.2f} | pNANO_v2 | {freq_nano:<12.2f} | {erro_nano:<17.2f} |")
print("=" * 60)


# --- Saída Gráfica (Visualização) ---
plt.figure(figsize=(12, 6))
plt.title(f'Análise de Frequência (FFT) - Teste de {GROUND_TRUTH_FREQ} Hz')
plt.plot(xf_uno, yf_uno, 'r-', label=f'pUNO (Pico: {freq_uno:.2f} Hz)', alpha=0.8)
plt.plot(xf_nano, yf_nano, 'b-', label=f'pNANO (Pico: {freq_nano:.2f} Hz)', alpha=0.8)
plt.axvline(x=GROUND_TRUTH_FREQ, color='k', linestyle='--', label=f'Ground Truth ({GROUND_TRUTH_FREQ} Hz)')

# Limita o eixo X para focar na área de interesse (ex: 0 a 10 Hz)
plt.xlim(0, 10) 
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude Normalizada')
plt.legend()
plt.grid(True)
plt.show()