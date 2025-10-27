# ==================================/ Análise de Estabilidade /=========================================                                                     
#     - Autora: Manuele S. Christofalo
#     - Data: 30/09/2025
#     - Feito para o TCC "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE SISTEMAS EMBARCADOS
#      PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON "
#
#     -> Após os experimentos, esse código analisa os dados obtidos em termos de estabilidade
#       (plataforma parada) para encontrar possíveis ruídos ou lags atrelados aos sensores.
#  ===================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- PARÂMETROS DO TESTE ---
ARQUIVO_UNO = 'data/pUNO_TesteC_1.CSV'
ARQUIVO_NANO = 'data/pNANO_TesteC_1.CSV'


# 1. Carrega e prepara os dados -----------------------------------------------
def carregar_dados_estaticos(arquivo):
    """Carrega o CSV, renomeia colunas e converte o tempo para minutos."""
    
    df = pd.read_csv(arquivo, header=0, on_bad_lines='skip')
    
    # Renomeia as colunas
    df.rename(columns={
        'Roll (x)': 'roll',
        ' Pitch (Y)': 'pitch',
        ' Yaw (Z)': 'yaw',
        ' Time (s)': 'tempo_s'
    }, inplace=True)
    
    
    # Converte tempo de segundos para minutos
    df['tempo_min'] = df['tempo_s'] / 60.0 
    return df


# 2. Execução Principal -------------------------------------------------------
print("--- Script 3: Análise de Estabilidade (Ruído e Deriva) ---")

df_uno = carregar_dados_estaticos(ARQUIVO_UNO)
df_nano = carregar_dados_estaticos(ARQUIVO_NANO)


# A- Análise de Ruído (Desvio Padrão) ---
ruido_uno = df_uno[['roll', 'pitch', 'yaw']].std()
ruido_nano = df_nano[['roll', 'pitch', 'yaw']].std()

print("\n--- Tabela 1: Análise de Ruído (Desvio Padrão em Graus) ---")
print("=" * 40)
print(f"| Eixo  | pUNO_v2 | pNANO_v2 |")
print("-" * 40)
print(f"| Roll  | {ruido_uno['roll']:<7.4f} | {ruido_nano['roll']:<8.4f} |")
print(f"| Pitch | {ruido_uno['pitch']:<7.4f} | {ruido_nano['pitch']:<8.4f} |")
print(f"| Yaw   | {ruido_uno['yaw']:<7.4f} | {ruido_nano['yaw']:<8.4f} |")
print("=" * 40)
print("(Valores menores indicam sinal mais limpo/menos ruidoso)")


# B- Análise de Deriva (Drift) ---
# O 'Yaw' é o eixo que mais sofre com a deriva do giroscópio.
# Vamos normalizar ambos para começar em 0.
df_uno['yaw_drift'] = df_uno['yaw'] - df_uno['yaw'].iloc[0]
df_nano['yaw_drift'] = df_nano['yaw'] - df_nano['yaw'].iloc[0]

deriva_total_uno = df_uno['yaw_drift'].iloc[-1]
deriva_total_nano = df_nano['yaw_drift'].iloc[-1]

print("\n--- Tabela 2: Análise de Deriva (Drift) Total ---")
print("=" * 55)
print(f"| Métrica                 | pUNO_v2         | pNANO_v2        |")
print("-" * 55)
print(f"| Deriva Total em 'Yaw' | {deriva_total_uno:<15.2f} | {deriva_total_nano:<15.2f} |")
print("=" * 55)
print(f"(Valores próximos de 0 indicam maior estabilidade)")

plt.figure(figsize=(12, 6))
plt.title('Análise de Deriva (Drift) do Eixo "Yaw" em Repouso')
plt.plot(df_uno['tempo_min'], df_uno['yaw_drift'], 'r-', label='pUNO_v2 (Sem Magnetômetro)')
plt.plot(df_nano['tempo_min'], df_nano['yaw_drift'], 'b-', label='pNANO_v2 (Com Magnetômetro e Filtro de Kalman)')
plt.xlabel('Tempo (minutos)')
plt.ylabel('Deriva Acumulada em "Yaw" (Graus)')
plt.legend()
plt.grid(True)
plt.show()