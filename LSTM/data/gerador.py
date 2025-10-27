import pandas as pd
import numpy as np

# --- Configurações ---
N_COLETAS = 10
DURACAO_S = 3600  # 1 hora
AMOSTRAS_POR_S = 10
N_LINHAS_POR_COLETA = DURACAO_S * AMOSTRAS_POR_S  # 36.000
ARQUIVO_SAIDA = 'exemplo_artificial.csv'

def gerar_sinal_coleta(n_linhas: int, time_s: np.ndarray) -> tuple:
    """Gera dados de Roll, Pitch, Yaw e Tremor para uma coleta."""
    
    # 1. Gerar a flag de Tremor
    # Começa com tudo 0 (Não-Tremor)
    tremor_flag = np.zeros(n_linhas, dtype=int)
    
    # Adicionar períodos aleatórios de tremor
    n_periodos_tremor = np.random.randint(20, 40) # 20 a 40 eventos de tremor por hora
    
    for _ in range(n_periodos_tremor):
        # Duração do tremor: 10 a 60 segundos
        duracao_tremor_s = np.random.randint(10, 61)
        # Início do tremor
        inicio_tremor_s = np.random.randint(0, DURACAO_S - duracao_tremor_s)
        
        # Converter para índices
        idx_inicio = inicio_tremor_s * AMOSTRAS_POR_S
        idx_fim = (inicio_tremor_s + duracao_tremor_s) * AMOSTRAS_POR_S
        
        tremor_flag[idx_inicio:idx_fim] = 1
        
    # 2. Gerar Sinais (Roll, Pitch, Yaw)
    
    # Gerar sinal de base (movimento lento, "normal")
    base_roll = 0.5 * np.sin(2 * np.pi * 0.05 * time_s) + np.cumsum(np.random.normal(0, 0.005, n_linhas))
    base_pitch = 0.3 * np.sin(2 * np.pi * 0.08 * time_s) + np.cumsum(np.random.normal(0, 0.005, n_linhas))
    base_yaw = 0.4 * np.sin(2 * np.pi * 0.03 * time_s) + np.cumsum(np.random.normal(0, 0.005, n_linhas))
    
    # Gerar oscilação de tremor (alta frequência, 4-6 Hz)
    # A amplitude também varia um pouco
    amp_roll = np.random.uniform(1.0, 2.0)
    amp_pitch = np.random.uniform(0.8, 1.8)
    amp_yaw = np.random.uniform(0.5, 1.5)
    
    freq_roll = np.random.uniform(4.0, 6.0)
    freq_pitch = np.random.uniform(4.0, 6.0)
    freq_yaw = np.random.uniform(4.0, 6.0)

    # O sinal de tremor só existe onde a flag é 1
    sinal_tremor_roll = amp_roll * np.sin(2 * np.pi * freq_roll * time_s) * tremor_flag
    sinal_tremor_pitch = amp_pitch * np.sin(2 * np.pi * freq_pitch * time_s) * tremor_flag
    sinal_tremor_yaw = amp_yaw * np.sin(2 * np.pi * freq_yaw * time_s) * tremor_flag
    
    # 3. Sinal final = Base + Tremor
    roll = base_roll + sinal_tremor_roll
    pitch = base_pitch + sinal_tremor_pitch
    yaw = base_yaw + sinal_tremor_yaw
    
    return roll, pitch, yaw, tremor_flag

def main():
    print(f"Gerando arquivo de mock '{ARQUIVO_SAIDA}'...")
    print(f"Configuração: {N_COLETAS} coletas, {N_LINHAS_POR_COLETA} linhas/coleta.")
    
    lista_dfs = []
    
    for i in range(1, N_COLETAS + 1):
        print(f"Processando Coleta {i}/{N_COLETAS}...")
        
        # Criar colunas base
        id_coleta = np.full(N_LINHAS_POR_COLETA, i)
        time_s = np.linspace(0, DURACAO_S, N_LINHAS_POR_COLETA)
        
        # Gerar sinais
        roll, pitch, yaw, tremor = gerar_sinal_coleta(N_LINHAS_POR_COLETA, time_s)
        
        # Montar DataFrame
        df_coleta = pd.DataFrame({
            'ID_Coleta': id_coleta,
            'Roll (x)': roll,
            'Pitch (y)': pitch,
            'Yaw (z)': yaw,
            'Time (s)': time_s,
            'Tremor': tremor
        })
        lista_dfs.append(df_coleta)
        
    print("Combinando todas as coletas...")
    df_final = pd.concat(lista_dfs)
    
    # Arredondar para 4 casas decimais para economizar espaço
    df_final = df_final.round({'Roll (x)': 4, 'Pitch (y)': 4, 'Yaw (z)': 4, 'Time (s)': 1})
    
    print(f"Salvando arquivo em '{ARQUIVO_SAIDA}'...")
    df_final.to_csv(ARQUIVO_SAIDA, index=False)
    
    print("\nArquivo gerado com sucesso!")
    print(f"Total de linhas: {len(df_final)}")
    print("\nDistribuição das classes (Tremor):")
    print(df_final['Tremor'].value_counts(normalize=True))

if __name__ == "__main__":
    # Verifique se você tem pandas e numpy instalados:
    # pip install pandas numpy
    main()
