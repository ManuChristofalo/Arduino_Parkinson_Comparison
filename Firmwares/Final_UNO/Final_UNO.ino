/* ================================/  Codigo Final - UNO /=====================================                                                     
    - Autora: Manuele S. Christofalo
     - Data: 25/08/2025
     - Feito para o TCC: "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE SISTEMAS EMBARCADOS
     PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON "
 ===================================================================================================*/

// 1. DECLARACAO DAS VARIAVEIS BASE ---------------------------------------------------------------------------------------------
#include <Wire.h>
#include <MPU6050.h>
#include <SD.h>

MPU6050 mpu;                                                                // Declaracao da MPU (acc + gyro)

float AccX, AccY, AccZ;                                                     // Valores lidos pelo acelerometro
float GyroX, GyroY, GyroZ;                                                  // Valores lidos pelo giroscopio

int16_t rawAccX, rawAccY, rawAccZ;
int16_t rawGyroX, rawGyroY, rawGyroZ;

int base = 0;                                                               // Zero do grafico

float tempoDelta, tempoInicial, tempoAtual, tempoAnterior, tempoSegundos;   // Variaveis de tempo para calculo do angulo (vel = a°/s)
float AnguloAccX, AnguloAccY;                                               // Angulos calculados para o acelerometro
float rotX, rotY, rotZ;                                                     // Angulo de rotacao total do braco (roll, pitch, yaw)

float kalmanEstadoX = 0, kalmanIncertezaX = 1;                              // Variaveis do filtro de Kalman para X (roll)
float kalmanEstadoY = 0, kalmanIncertezaY = 1;                              // Variaveis do filtro de Kalman para Y (pitch)
float kalmanEstadoZ = 0, kalmanIncertezaZ = 1;                              // Variaveis do filtro de Kalman para Z (yaw)

float varGyro = 0.0025;                                                     // Variancia calculada com base na doc oficial do Arduino para o Gyro
float varAcc = 0.000016;                                                    // Variancia calculada com base na doc oficial do Arduino para o Acc

File arquivo;                                                               // Variavel arquivo
String nomeArquivo = "";                                                    // Variavel do nome do arquivo
String blocoRegistro = "";                                                  // Armazena 10 linhas antes de gravar
int count = 0;                                                              // Conta quantas amostras temos no buffer


// 2. FUNÇOES CRIADAS ----------------------------------------------------------------------------------------------------------
void kalman(float *kalmanEstimado, float *kalmanIncerteza, float taxa, float medida, float tempoDelta, float varAccGyro){
    //I. Predicao do estado: estimativa de onde o angulo estaria se so a entrada fosse considerada
    *kalmanEstimado = *kalmanEstimado + tempoDelta * taxa;

    //II. Predicao da incerteza: a incerteza cresce proporcionalmente ao tempo e ao ruido assumido do processo
    *kalmanIncerteza = *kalmanIncerteza + tempoDelta * tempoDelta * varGyro;

    //III. Calculo do ganho de Kalman: o ganho pondera quanto confiar na previsão vs na medida
    float kalmanGain = *kalmanIncerteza / (*kalmanIncerteza + varAccGyro); // varAccMAg: variancia calculada de acordo com doc oficial do Arduino para Acc ou Gyro

    //IV. Correcao da estimativa e da incerteza
    *kalmanEstimado = *kalmanEstimado + kalmanGain * (medida - *kalmanEstimado);
    *kalmanIncerteza = (1 - kalmanGain) * (*kalmanIncerteza);
}


String NomeArquivo(){
    int i=1;
    String nome = "TCCuno1.csv";

    while(SD.exists(nome)){
        i++;
        nome = "TCCuno" + String(i) + ".csv";
    }

    return nome;
}


// 3. INICIALIZACAO DO SENSOR  ---------------------------------------------------------------------------------------
void setup() {
    //Inicializacao do MPU (acc + gyro)
    Wire.begin();
    mpu.initialize();
    if (!mpu.testConnection()) {
        while (1);
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

    //Abertura arquivo com a garantia que ele existe
    arquivo = SD.open(nomeArquivo, FILE_WRITE);
    //Se o arquivo abriu, escreve:
    if(arquivo){
        arquivo.print("Roll (x), Pitch (Y), Yaw (Z), Time (s)\n");
        arquivo.close();
    }

    tempoAtual = millis();
    tempoInicial = tempoAtual;
    tempoAnterior = tempoAtual;
}


// 4. LEITURAS -------------------------------------------------------------------------------------------------------------------
void loop() {
    // 4.1. Leitura do tempo ====
    tempoAnterior = tempoAtual;
    tempoAtual = millis();
    tempoDelta = (tempoAtual - tempoAnterior) / 1000.0;


    // 4.2. Leitura e ajuste do acelerometro ====
    mpu.getAcceleration(&rawAccX, &rawAccY, &rawAccZ);

    //O valor bruto foi convertido com base na escala padrao do MPU6050 (16g -> 16384 LSB/g)
    AccX = rawAccX / 16384.0;
    AccY = rawAccY / 16384.0;
    AccZ = rawAccZ / 16384.0;


    // 4.3. Leitura e ajuste do giroscopio ====
    mpu.getRotation(&rawGyroX, &rawGyroY, &rawGyroZ);

    //A unidade lida é em "raw", e 131 LSB/°/s corresponde a sensibilidade padrao para ±250°/s
    GyroX = rawGyroX / 131.0;
    GyroY = rawGyroY / 131.0;
    GyroZ = rawGyroZ / 131.0;


    // 4.4. Aplicacao do Filtro de Kalman ====
    AnguloAccX = atan2(AccY, sqrt(AccX * AccX + AccZ * AccZ)) * 180.0 / PI;
    AnguloAccY = atan2(-AccX, sqrt(AccY * AccY + AccZ * AccZ)) * 180.0 / PI;

    kalman(&kalmanEstadoX, &kalmanIncertezaX, GyroX, AnguloAccX, tempoDelta, varAcc); // Roll
    kalman(&kalmanEstadoY, &kalmanIncertezaY, GyroY, AnguloAccY, tempoDelta, varAcc); // Pitch
    kalman(&kalmanEstadoZ, &kalmanIncertezaZ, GyroZ, GyroZ, tempoDelta, varGyro); //yaw


    // 4.5. Resultado final e impressao ====
    rotX = kalmanEstadoX;
    rotY = kalmanEstadoY;
    rotZ = kalmanEstadoZ;

    // Adiciona a linha ao bloco de registro
    tempoSegundos = (tempoAtual - tempoInicial) / 1000; // -9000 pois eh o tempo medio para o Arduino comecar a gravar
    blocoRegistro += String(rotX) + "," + String(rotY) + "," + String(rotZ) + "," + String(tempoSegundos, 2) + "\n";
    count++;

    // A gravacao do bloco no arquivo eh feita a cada 10 iteracoes (aprox 1s)
    if(count == 10){
        arquivo = SD.open(nomeArquivo, FILE_WRITE);

        //Se o arquivo abriu, escreve o bloco de 1s de registro
        if(arquivo){
            arquivo.print(blocoRegistro);
            arquivo.close();
        }
        else{
            while(1);
        }

        // Limpa/reinicia as variaveis
        blocoRegistro = "";
        count = 0;
    }

    delay(100); // Pequeno atraso para estabilizar a taxa (~50 Hz)
}
