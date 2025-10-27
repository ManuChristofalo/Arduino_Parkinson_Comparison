/* ================================/ Codigo Final - Nano /=====================================                                                     
    - Autora: Manuele S. Christofalo
     - Data: 05/08/2025
     - Feito para o TCC "ANÁLISE COMPARATIVA DE ARDUINOS NA IMPLEMENTAÇÃO DE SISTEMAS EMBARCADOS
     PARA MONITORAMENTO DE TREMORES NA DOENÇA DE PARKINSON "
 ===================================================================================================*/

// 1. DECLARACAO DAS VARIAVEIS BASE ---------------------------------------------------------------------------------------------
#include "Arduino_BMI270_BMM150.h"
#include <SPI.h>
#include <SD.h>

float AccX, AccY, AccZ;                                                     // Valores lidos pelo acelerometro
float GyroX, GyroY, GyroZ;                                                  // Valores lidos pelo giroscopio
float MagX, MagY, MagZ;                                                     // Valores lidos pelo magnetômetro

int base = 0;                                                               // Zero do grafico

float tempoDelta, tempoInicial, tempoAtual, tempoAnterior, tempoSegundos;   // Variaveis de tempo para calculo do angulo (vel = a°/s)
float AnguloAccX, AnguloAccY;                                               // Angulos calculados para o acelerometro
float AnguloMagZ;                                                           // Angulo do campo magnetico terrestre (Yaw)
float rotX, rotY, rotZ;                                                     // Angulo de rotacao total do braco (roll, pitch, yaw)

float kalmanEstadoX = 0, kalmanIncertezaX = 1;                              // Variaveis do filtro de Kalman para X (roll)
float kalmanEstadoY = 0, kalmanIncertezaY = 1;                              // Variaveis do filtro de Kalman para Y (pitch)
float kalmanEstadoZ = 0, kalmanIncertezaZ = 1;                              // Variaveis do filtro de Kalman para Z (yaw)

float varGyro = 0.0016333333;                                               // Variancia calculada com base na doc oficial do Arduino para o Gyro
float varAcc = 0.0000000166;                                                // Variancia calculada com base na doc oficial do Arduino para o Acc
float varMag = 0.0000653333;                                                // Variancia calculada com base na doc oficial do Arduino para o Mag

File arquivo;                                                               // Variavel arquivo
String nomeArquivo = "";                                                    // Variavel do nome do arquivo
String blocoRegistro = "";                                                  // Armazena 10 linhas antes de gravar
int count = 0;                                                              // Conta quantas amostras temos no buffer


// 2. FUNÇOES CRIADAS ----------------------------------------------------------------------------------------------------------
void kalman(float *kalmanEstimado, float *kalmanIncerteza, float taxa, float medida, float tempoDelta, float varAccMag){
    //I. Predicao do estado: estimativa de onde o angulo estaria se so a entrada fosse considerada
    *kalmanEstimado = *kalmanEstimado + tempoDelta * taxa;

    //II. Predicao da incerteza: a incerteza cresce proporcionalmente ao tempo e ao ruido assumido do processo
    *kalmanIncerteza = *kalmanIncerteza + tempoDelta * tempoDelta * varGyro;

    //III. Calculo do ganho de Kalman: o ganho pondera quanto confiar na previsão vs na medida
    float kalmanGain = *kalmanIncerteza / (*kalmanIncerteza + varAccMag); // varAccMAg: variancia calculada de acordo com doc oficial do Arduino para Acc ou Mag

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


// 3. INICIALIZACAO DO SENSOR  ---------------------------------------------------------------------------------------
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
    if(IMU.accelerationAvailable()){
        IMU.readAcceleration(AccX, AccY, AccZ);

        // Calculo dos angulos (roll e pitch) em graus
        AnguloAccX = atan2(AccY, sqrt(AccX * AccX + AccZ * AccZ)) * 180.0 / PI;
        AnguloAccY = atan2(-AccX, sqrt(AccY * AccY + AccZ * AccZ)) * 180.0 / PI;
    }


    // 4.3. Leitura e ajuste do giroscopio ====
    if(IMU.gyroscopeAvailable()){
        IMU.readGyroscope(GyroX, GyroY, GyroZ);

        // Sensor retorna em rad/s: converter para graus/s
        GyroX *= 180.0 / PI;
        GyroY *= 180.0 / PI;
        GyroZ *= 180.0 / PI;
    }


    // 4.4. Leitura do magnetômetro (Yaw absoluto) ====
    if(IMU.magneticFieldAvailable()){
        IMU.readMagneticField(MagX, MagY, MagZ);

        // Calculo do angulo de Yaw absoluto (com base na Terra) em graus
        AnguloMagZ = atan2(MagY, MagX) * 180.0 / PI;

        // Correcao de angulos negativos para faixa 0°–360°
        if (AnguloMagZ < 0) AnguloMagZ += 360.0;
    }


    // 4.5. Aplicacao do Filtro de Kalman ====
    kalman(&kalmanEstadoX, &kalmanIncertezaX, GyroX, AnguloAccX, tempoDelta, varAcc); // Roll
    kalman(&kalmanEstadoY, &kalmanIncertezaY, GyroY, AnguloAccY, tempoDelta, varAcc); // Pitch
    kalman(&kalmanEstadoZ, &kalmanIncertezaZ, GyroZ, AnguloMagZ, tempoDelta, varMag); // Yaw (corrigido com magnetometro)


    // 4.6. Resultado final e impressao ====
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
