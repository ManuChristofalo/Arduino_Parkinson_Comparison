/* ================================/ Codigo Final - Nano + TinyML /=====================================
    - Autora: Manuele S. Christofalo
    - Data: 25/10/2025
    - Feito para o TCC "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE SISTEMAS EMBARCADOS
      PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON "
    
    - Este firmware combina a coleta de dados da IMU com Filtro de Kalman 
    e a inferência em tempo real usando um modelo LSTM treinado (TinyML).
 ===================================================================================================*/

 
 // Bibliotecas Originais
 #include "Arduino_BMI270_BMM150.h"
 #include <SPI.h>
 #include <SD.h>
 
 // Bibliotecas TinyML
 #include <TensorFlowLite.h>
 #include "model_data.h" // Inclui o arquivo do modelo (criado com 'xxd')
 
 #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
 #include "tensorflow/lite/micro/micro_interpreter.h"
 
 
 // 1. DECLARACAO DAS VARIAVEIS BASE ---------------------------------------------------------------------------------------------
// Variáveis Originais de Sensores
float AccX, AccY, AccZ;     // Valores lidos pelo acelerometro
float GyroX, GyroY, GyroZ;  // Valores lidos pelo giroscopio
float MagX, MagY, MagZ;     // Valores lidos pelo magnetômetro
int base = 0;               // Zero do grafico

// Variáveis Originais de Tempo
float tempoDelta, tempoInicial, tempoAtual, tempoAnterior, tempoSegundos;
float AnguloAccX, AnguloAccY; // Angulos calculados para o acelerometro
float AnguloMagZ;             // Angulo do campo magnetico terrestre (Yaw)
float rotX, rotY, rotZ;       // Angulo de rotacao total do braco (roll, pitch, yaw)

// Variáveis Originais do Filtro de Kalman
float kalmanEstadoX = 0, kalmanIncertezaX = 1; // Variaveis do filtro de Kalman para X (roll)
float kalmanEstadoY = 0, kalmanIncertezaY = 1; // Variaveis do filtro de Kalman para Y (pitch)
float kalmanEstadoZ = 0, kalmanIncertezaZ = 1; // Variaveis do filtro de Kalman para Z (yaw)

float varGyro = 0.0016333333; // Variancia calculada com base na doc oficial do Arduino para o Gyro
float varAcc = 0.0000000166;  // Variancia calculada com base na doc oficial do Arduino para o Acc
float varMag = 0.0000653333;  // Variancia calculada com base na doc oficial do Arduino para o Mag

// Variáveis Originais do SD Card [cite: 14-15]
File arquivo;               // Variavel arquivo
String nomeArquivo = "";    // Variavel do nome do arquivo
String blocoRegistro = "";  // Armazena 10 linhas antes de gravar
int count = 0;              // Conta quantas amostras temos no buffer


// Variáveis do Modelo (baseado em main.py)
const int WINDOW_SIZE = 50; // 50 amostras (5 segundos * 10 Hz)
const int N_FEATURES = 4;   // Roll (x), Pitch (y), Yaw (z), Magnitude
const int STEP_SIZE = 10;   // Executa a inferência a cada 10 amostras (1 segundo)


float g_input_buffer[WINDOW_SIZE * N_FEATURES]; // Buffer de entrada para o modelo (Janela Deslizante)

// Variáveis do TensorFlow Lite
namespace {
  tflite::ErrorReporter* g_error_reporter = nullptr;
  const tflite::Model* g_model = nullptr;
  tflite::MicroInterpreter* g_interpreter = nullptr;
  TfLiteTensor* g_model_input = nullptr;    // Ponteiro para o tensor de entrada
  TfLiteTensor* g_model_output = nullptr;   // Ponteiro para o tensor de saída

  // Arena de Tensores: Memória principal para o TFLM
  constexpr int kTensorArenaSize = 12 * 1024; // 12KB (LSTMs usam mais RAM)
  uint8_t g_tensor_arena[kTensorArenaSize];
}


// Constantes de Normalização (Scaler) - AÇÃO NECESSÁRIA!
// Substitua os valores pela MÉDIA (MEANS) e ESCALA (SCALES) gerados pelo script de treinamento
// FEATURES = ['Roll (x)', 'Pitch (y)', 'Yaw (z)', 'Magnitude']
const float SCALER_MEANS[N_FEATURES]   = { 0.0, 0.0, 0.0, 0.0 };
const float SCALER_SCALES[N_FEATURES] = { 1.0, 1.0, 1.0, 1.0 };


// 2. FUNÇOES CRIADAS ----------------------------------------------------------------------------------------------------------

void kalman(float *kalmanEstimado, float *kalmanIncerteza, float taxa, float medida, float tempoDelta, float varAccMag){
    //I. Predicao do estado: estimativa de onde o angulo estaria se so a entrada fosse considerada
    *kalmanEstimado = *kalmanEstimado + tempoDelta * taxa;
    //II. Predicao da incerteza: a incerteza cresce proporcionalmente ao tempo e ao ruido assumido do processo
    *kalmanIncerteza = *kalmanIncerteza + tempoDelta * tempoDelta * varGyro;
    //III. Calculo do ganho de Kalman: o ganho pondera quanto confiar na previsão vs na medida
    float kalmanGain = *kalmanIncerteza / (*kalmanIncerteza + varAccMag);
    //IV. Correcao da estimativa e da incerteza
    *kalmanEstimado = *kalmanEstimado + kalmanGain * (medida - *kalmanEstimado);
    *kalmanIncerteza = (1 - kalmanGain) * (*kalmanIncerteza);
}


String NomeArquivo(){
    int i=1;
    String nome = "TCC1.csv";
    while(SD.exists(nome)){
        i++;
        nome = "TCC" + String(i) + ".csv";
    }
    return nome;
}


// 3. INICIALIZACAO DO SENSOR E DO MODELO  -------------------------------------------------------------------------------------
void setup() {
    //Inicializacao da IMU (acc + gyro + mag)
    if(!IMU.begin()){
        while(1);
    }

    //Abertura SD
    if(!SD.begin(10)){
        while(1);
    }
    
    //Abertura arquivo
    nomeArquivo = NomeArquivo();
    arquivo = SD.open(nomeArquivo, FILE_WRITE);
    arquivo.close();
    delay(2000); // Delay que garante a abertura
    arquivo = SD.open(nomeArquivo, FILE_WRITE);
    if(arquivo){
        // Cabeçalho modificado para incluir Magnitude e Predição
        arquivo.print("Roll (x), Pitch (Y), Yaw (Z), Magnitude, Time (s), Prediction, Probability\n");
        arquivo.close();
    }

    // Inicialização do tempo
    tempoAtual = millis();
    tempoInicial = tempoAtual;
    tempoAnterior = tempoAtual;

    
    // Inicialização do TensorFlow Lite 
    memset(g_input_buffer, 0, sizeof(g_input_buffer)); // Inicializa o buffer de entrada com zeros

    static tflite::MicroErrorReporter micro_error_reporter; // Configura o Error Reporter
    g_error_reporter = &micro_error_reporter;
    g_error_reporter->Report("Iniciando o TensorFlow Lite...");

    g_model = tflite::GetModel(g_tflite_model_data); // Carrega o modelo (model_data.h)
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        g_error_reporter->Report("Versão do modelo incompatível!");
        while(1);
    }

    // Resolve as operações (Layers) que o modelo usa
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddUnidirectionalSequenceLSTM();
    resolver.AddFullyConnected();
    resolver.AddLogistic(); // (Camada de ativação 'sigmoid')
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddSin(); // Pode ser usado por otimizações internas da LSTM
    resolver.AddCos(); // Pode ser usado por otimizações internas da LSTM
    
    // Instancia o Interpretador
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize, g_error_reporter
    );
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        g_error_reporter->Report("Falha ao alocar tensores! Aumente a kTensorArenaSize.");
        while(1);
    }

    // Tensores de entrada e saída
    g_model_input = g_interpreter->input(0);
    g_model_output = g_interpreter->output(0);
    
    g_error_reporter->Report("TFLM Inicializado com sucesso.");
}


// 4. LOOP PRINCIPAL (LEITURA, FILTRO, INFERÊNCIA E GRAVAÇÃO) ------------------------------------------------------------------
void loop() {
    
    // 4.1. Leitura do tempo
    tempoAnterior = tempoAtual;
    tempoAtual = millis();
    tempoDelta = (tempoAtual - tempoAnterior) / 1000.0;
    
    // 4.2. Leitura e ajuste do acelerometro
    if(IMU.accelerationAvailable()){
        IMU.readAcceleration(AccX, AccY, AccZ);
        AnguloAccX = atan2(AccY, sqrt(AccX * AccX + AccZ * AccZ)) * 180.0 / PI;
        AnguloAccY = atan2(-AccX, sqrt(AccY * AccY + AccZ * AccZ)) * 180.0 / PI;
    }

    // 4.3. Leitura e ajuste do giroscopio
    if(IMU.gyroscopeAvailable()){
        IMU.readGyroscope(GyroX, GyroY, GyroZ);
        GyroX *= 180.0 / PI;
        GyroY *= 180.0 / PI;
        GyroZ *= 180.0 / PI;
    }

    // 4.4. Leitura do magnetômetro (Yaw absoluto)
    if(IMU.magneticFieldAvailable()){
        IMU.readMagneticField(MagX, MagY, MagZ);
        AnguloMagZ = atan2(MagY, MagX) * 180.0 / PI;
        if (AnguloMagZ < 0) AnguloMagZ += 360.0;
    }

    // 4.5. Aplicacao do Filtro de Kalman
    kalman(&kalmanEstadoX, &kalmanIncertezaX, GyroX, AnguloAccX, tempoDelta, varAcc); // Roll
    kalman(&kalmanEstadoY, &kalmanIncertezaY, GyroY, AnguloAccY, tempoDelta, varAcc); // Pitch
    kalman(&kalmanEstadoZ, &kalmanIncertezaZ, GyroZ, AnguloMagZ, tempoDelta, varMag); // Yaw

    // 4.6. Resultado final (Roll, Pitch, Yaw)
    rotX = kalmanEstadoX;
    rotY = kalmanEstadoY;
    rotZ = kalmanEstadoZ;

    // 4.7 Pré-processamento e Buffer TinyML
    float mag = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ); // Magnitude

    // Normaliza os dados (StandardScaler)
    float features[N_FEATURES];
    features[0] = (rotX - SCALER_MEANS[0]) / SCALER_SCALES[0];
    features[1] = (rotY - SCALER_MEANS[1]) / SCALER_SCALES[1];
    features[2] = (rotZ - SCALER_MEANS[2]) / SCALER_SCALES[2];
    features[3] = (mag  - SCALER_MEANS[3]) / SCALER_SCALES[3];

    // Adiciona a amostra ao buffer deslizante
    memmove(&g_input_buffer[0], 
            &g_input_buffer[N_FEATURES], 
            (WINDOW_SIZE - 1) * N_FEATURES * sizeof(float));

    memcpy(&g_input_buffer[(WINDOW_SIZE - 1) * N_FEATURES], 
           features, 
           N_FEATURES * sizeof(float));

    // 4.8. Gravacao no SD e Inferência
    tempoSegundos = (tempoAtual - tempoInicial) / 1000.0;
    
    // Adiciona os dados ao bloco de registro (sem a predição ainda)
    blocoRegistro += String(rotX) + "," + String(rotY) + "," + String(rotZ) + "," + String(mag) + "," + String(tempoSegundos, 2);
    count++;

    // A gravacao do bloco E A INFERÊNCIA são feitas a cada 10 iteracoes (STEP_SIZE)
    if(count == STEP_SIZE){
        // Copia o buffer deslizante completo (50x4) para o tensor de entrada do TFLM
        for(int i = 0; i < (WINDOW_SIZE * N_FEATURES); i++) g_model_input->data.f[i] = g_input_buffer[i];

        // Executa a Inferência
        float probability = 0.0; // Probabilidade de tremor
        int prediction = 0;      // Classe final (0 ou 1)
        
        if(g_interpreter->Invoke() == kTfLiteOk){
            probability = g_model_output->data.f[0]; // Saída da sigmoid
            
            // Converte probabilidade em classe (0 ou 1)
            if(probability > 0.5) prediction = 1;
        }
        
        else{
            g_error_reporter->Report("Falha na Inferência!");
            probability = -1; // Indica erro
        }

        // Adiciona a predição ao bloco de registro
        blocoRegistro += "," + String(prediction) + "," + String(probability, 4) + "\n";

        // Gravacao do bloco no SD (Lógica original)
        arquivo = SD.open(nomeArquivo, FILE_WRITE);
        if(arquivo){
            arquivo.print(blocoRegistro);
            arquivo.close();
        }
        else{
            g_error_reporter->Report("Falha ao abrir SD Card para escrita!");
            while(1);
        }

        // Limpa/reinicia as variaveis
        blocoRegistro = "";
        count = 0;
    }
    
    // Se não for hora de gravar, apenas adiciona uma nova linha para a próxima amostra
    else blocoRegistro += ",,,\n"; 
    
    delay(100); // Pequeno atraso para estabilizar a taxa (~10 Hz)
}