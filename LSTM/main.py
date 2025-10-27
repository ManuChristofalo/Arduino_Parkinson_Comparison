# ==================================/ main.py /==========================================
#   - Feito por: Manuele Christófalo
#   - Aplicado na pesquisa: "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE
#      SISTEMAS EMBARCADOS PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON"
#
#   --> Função principal. Pipeline: normalizar, treinar, avaliar
# =======================================================================================

# Bibliotecas
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Módulos Locais
import data_loader
import preprocessing
import model as model_builder
import plotting

# 0. Configurações Principais -------------------------------------------------
CSV_PATH = 'data/exemplo_artificial.csv'  # -> alterar para csv desejado

# Divisão dos dados (8 para treino, 2 para teste)
TRAIN_COLETAS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_COLETAS = [9, 10]

# Features que a LSTM usará
FEATURES = ['Roll (x)', 'Pitch (y)', 'Yaw (z)', 'Magnitude']
TARGET_COL = 'Tremor'

# Configurações da Janela Deslizante
WINDOW_SIZE = 50    # 10 amostras/segundo * 5 segundos = 50 amostras
STEP = 10           # Desliza a janela em 1 segundo (10 amostras)

# Configurações do Modelo
EPOCHS = 20
BATCH_SIZE = 64


def main():
    print("Iniciando pipeline de detecção de tremor com LSTM...")

# 1. Carregamento e pré-processamento dos dados -------------------------------
    df = data_loader.load_data(CSV_PATH)
    if df.empty:
        return
    df = data_loader.add_features(df)
    
    # Divide em treino e teste ANTES de qualquer processamento 
    df_train, df_test = data_loader.split_data_by_coleta(df, TRAIN_COLETAS, TEST_COLETAS)
    
    # Normalização
    print("\n--- Etapa 1: Normalização ---")
    scaler = preprocessing.get_scaler(df_train, FEATURES)
    
    df_train_scaled = preprocessing.scale_data(df_train, scaler, FEATURES)
    df_test_scaled = preprocessing.scale_data(df_test, scaler, FEATURES)
    
    # Plotar dados normalizados (da primeira coleta de teste)
    primeira_coleta_teste = df_test_scaled[
        df_test_scaled['ID_Coleta'] == TEST_COLETAS[0]
    ]
    plotting.plot_normalized_data(primeira_coleta_teste, FEATURES, n_samples=2000)

    # Criação das Sequências (Janelas)
    print("Criando sequências (janelas) para treino...")
    X_train, y_train = preprocessing.create_sequences(
        df_train_scaled, FEATURES, TARGET_COL, WINDOW_SIZE, STEP
    )
    
    print("Criando sequências (janelas) para teste...")
    X_test, y_test = preprocessing.create_sequences(
        df_test_scaled, FEATURES, TARGET_COL, WINDOW_SIZE, STEP
    )
    
    print(f"Formato dos dados de treino (X): {X_train.shape}")
    print(f"Formato dos dados de treino (y): {y_train.shape}")
    print(f"Formato dos dados de teste (X): {X_test.shape}")
    print(f"Formato dos dados de teste (y): {y_test.shape}")

    if X_train.shape[0] == 0:
        print("Erro: Nenhuma sequência de treino foi criada. Verifique WINDOW_SIZE e os dados.")
        return


# 2. Treinamento do Modelo ----------------------------------------------------
    print("\n--- Etapa 2: Construção e Treinamento do Modelo ---")
    
    # Calcular pesos de classe (importante se houver desbalanceamento)
    # Se houver 90% de '0' e 10% de '1', isso ajuda o modelo.
    classes_unicas = np.unique(y_train)
    if len(classes_unicas) > 1:
        weights = compute_class_weight('balanced', classes=classes_unicas, y=y_train)
        class_weight = {cls: weight for cls, weight in zip(classes_unicas, weights)}
        print(f"Pesos de Classe calculados: {class_weight}")
    else:
        class_weight = None
        print("Apenas uma classe encontrada nos dados de treino. Não foi possível calcular pesos.")

    # Construir e compilar o modelo
    model = model_builder.build_model(WINDOW_SIZE, len(FEATURES))
    model = model_builder.compile_model(model)
    model.summary()
    
    # Treinar
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # Usa 20% dos dados de TREINO para validação interna
        class_weight=class_weight,
        verbose=1
    )
    
    # Plotar histórico de treino
    plotting.plot_training_history(history)
    
    # Salvar o modelo (opcional)
    model.save("parkinson_lstm_model.h5")
    print("Modelo salvo como 'parkinson_lstm_model.h5'")


# 3. Aplicação e Validação do Modelo ------------------------------------------
    print("\n--- Etapa 3: Avaliação nas Coletas de Teste ---")
    
    # Avaliação geral no conjunto de teste
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Avaliação no Conjunto de Teste (Coletas {TEST_COLETAS}):")
    print(f"  Perda (Loss): {loss:.4f}")
    print(f"  Acurácia:     {accuracy*100:.2f}%")
    
    # Fazer previsões
    y_pred_proba = model.predict(X_test)
    y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten() # Converte probabilidade em 0 ou 1
    
    # Relatório de Classificação Detalhado
    print("\nRelatório de Classificação (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred_classes, target_names=['0: Normal', '1: Tremor']))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred_classes))
    
    # Plotar previsões vs. realidade
    plotting.plot_predictions(
        y_test, 
        y_pred_classes, 
        title=f"Previsão vs. Realidade (Coletas de Teste {TEST_COLETAS})"
    )

if __name__ == "__main__":
    main()