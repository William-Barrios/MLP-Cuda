#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

const string training_image_fn = "train-images.idx3-ubyte";
const string training_label_fn = "train-labels.idx1-ubyte";
const string testing_image_fn = "t10k-images.idx3-ubyte";
const string testing_label_fn = "t10k-labels.idx1-ubyte";

const int nTraining = 60000;
const int nTest = 10000;
const int height = 28;
const int width = 28;
const int inputSize = (height * width) + 1;
const int hiddenSize = 128; // Número de neuronas en la capa oculta
const int outputSize = 10; // Número de neuronas en la capa de salida

ifstream image;
ifstream label;

vector<double> input(inputSize);
vector<double> hidden(hiddenSize);
vector<double> output(outputSize);
vector<double> weightsInputHidden(inputSize* hiddenSize, 0.0);
vector<double> weightsHiddenOutput(hiddenSize* outputSize, 0.0);

// Función sigmoide y su derivada
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// Kernel para la propagación hacia adelante
__global__ void forwardKernel(const double* d_input, const double* d_weightsInputHidden, double* d_hidden,
    const double* d_weightsHiddenOutput, double* d_output) {
    int i = threadIdx.x;

    // Propagación a la capa oculta
    __shared__ double sumHidden[128];
    sumHidden[i] = 0.0;
    for (int j = 0; j < inputSize; j++) {
        sumHidden[i] += d_input[j] * d_weightsInputHidden[j * hiddenSize + i];
    }
    __syncthreads();

    if (i < hiddenSize) {
        d_hidden[i] = sigmoid(sumHidden[i]);
    }
    __syncthreads();

    // Propagación a la capa de salida
    __shared__ double sumOutput[10];
    sumOutput[i] = 0.0;
    for (int j = 0; j < hiddenSize; j++) {
        sumOutput[i] += d_hidden[j] * d_weightsHiddenOutput[j * outputSize + i];
    }
    __syncthreads();

    if (i < outputSize) {
        d_output[i] = sigmoid(sumOutput[i]);
    }
}

// Kernel para la retropropagación
__global__ void backwardKernel(const double* d_input, double* d_weightsInputHidden, double* d_hidden,
    double* d_weightsHiddenOutput, double* d_output, const double* d_labels) {
    int i = threadIdx.x;

    // Calcular el error de la capa de salida
    __shared__ double outputError[10];
    if (i < outputSize) {
        outputError[i] = (d_labels[i] - d_output[i]) * sigmoidDerivative(d_output[i]);
    }
    __syncthreads();

    // Calcular el error de la capa oculta
    __shared__ double hiddenError[128];
    if (i < hiddenSize) {
        hiddenError[i] = 0.0;
        for (int j = 0; j < outputSize; j++) {
            hiddenError[i] += outputError[j] * d_weightsHiddenOutput[i * outputSize + j];
        }
        hiddenError[i] *= sigmoidDerivative(d_hidden[i]);
    }
    __syncthreads();

    // Actualizar los pesos entre la capa oculta y la capa de salida
    if (i < outputSize) {
        for (int j = 0; j < hiddenSize; j++) {
            d_weightsHiddenOutput[j * outputSize + i] += 0.5 * d_hidden[j] * outputError[i];
        }
    }
    __syncthreads();

    // Actualizar los pesos entre la capa de entrada y la capa oculta
    if (i < hiddenSize) {
        for (int j = 0; j < inputSize; j++) {
            d_weightsInputHidden[j * hiddenSize + i] += 0.5 * d_input[j] * hiddenError[i];
        }
    }
}

// Función para inicializar los pesos aleatoriamente
void initializeWeights(vector<double>& weights) {
    for (auto& weight : weights) {
        weight = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

// Función para cargar datos del archivo MNIST
void loadMNISTData(const string& imageFile, const string& labelFile, vector<vector<double>>& images, vector<vector<double>>& labels, int nSamples) {
    ifstream imageStream(imageFile, ios::binary);
    ifstream labelStream(labelFile, ios::binary);

    if (!imageStream.is_open() || !labelStream.is_open()) {
        cerr << "No se pudo abrir el archivo de imagen o etiqueta." << endl;
        return;
    }

    // Saltar los encabezados
    imageStream.seekg(16);
    labelStream.seekg(8);

    for (int i = 0; i < nSamples; ++i) {
        images[i][0] = 1.0; // Bias
        for (int j = 1; j < inputSize; ++j) {
            unsigned char pixel = 0;
            imageStream.read((char*)&pixel, sizeof(pixel));
            images[i][j] = (pixel > 0) ? 1.0 : 0.0;
        }
        unsigned char label = 0;
        labelStream.read((char*)&label, sizeof(label));
        labels[i][label] = 1.0;
    }

    imageStream.close();
    labelStream.close();
}


// Función para probar la red
int testing() {
    vector<vector<double>> images(nTest, vector<double>(inputSize, 0.0));
    vector<vector<double>> labels(nTest, vector<double>(outputSize, 0.0));
    loadMNISTData(testing_image_fn, testing_label_fn, images, labels, nTest);

    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_weightsInputHidden;
    double* d_weightsHiddenOutput;

    cudaMalloc((void**)&d_input, inputSize * sizeof(double));
    cudaMalloc((void**)&d_hidden, hiddenSize * sizeof(double));
    cudaMalloc((void**)&d_output, outputSize * sizeof(double));
    cudaMalloc((void**)&d_weightsInputHidden, weightsInputHidden.size() * sizeof(double));
    cudaMalloc((void**)&d_weightsHiddenOutput, weightsHiddenOutput.size() * sizeof(double));

    int errors = 0;

    for (int i = 0; i < nTest; ++i) {
        cudaMemcpy(d_input, images[i].data(), inputSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weightsInputHidden, weightsInputHidden.data(), weightsInputHidden.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weightsHiddenOutput, weightsHiddenOutput.data(), weightsHiddenOutput.size() * sizeof(double), cudaMemcpyHostToDevice);

        forwardKernel << <1, max(hiddenSize, outputSize) >> > (d_input, d_weightsInputHidden, d_hidden, d_weightsHiddenOutput, d_output);
        cudaDeviceSynchronize();
        cudaMemcpy(output.data(), d_output, outputSize * sizeof(double), cudaMemcpyDeviceToHost);

        int predicted = distance(output.begin(), max_element(output.begin(), output.end()));
        int actual = distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end()));

        if (predicted != actual) {
            errors++;
        }
    }

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weightsInputHidden);
    cudaFree(d_weightsHiddenOutput);

    return errors;
}

// Función para entrenar la red
void trainNetwork(vector<vector<double>>& images, vector<vector<double>>& labels) {
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_weightsInputHidden;
    double* d_weightsHiddenOutput;
    double* d_labels;

    cudaMalloc((void**)&d_input, inputSize * sizeof(double));
    cudaMalloc((void**)&d_hidden, hiddenSize * sizeof(double));
    cudaMalloc((void**)&d_output, outputSize * sizeof(double));
    cudaMalloc((void**)&d_weightsInputHidden, weightsInputHidden.size() * sizeof(double));
    cudaMalloc((void**)&d_weightsHiddenOutput, weightsHiddenOutput.size() * sizeof(double));
    cudaMalloc((void**)&d_labels, outputSize * sizeof(double));

    std::ofstream outFileAccuracy("accuracy_data.txt");
    if (!outFileAccuracy) {
        cerr << "No se pudo abrir el archivo de precisión para escribir." << endl;
    }

    for (int epoch = 0; epoch < 5; ++epoch) { // Número de épocas
        for (int i = 0; i < images.size(); ++i) {
            cudaMemcpy(d_input, images[i].data(), inputSize * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weightsInputHidden, weightsInputHidden.data(), weightsInputHidden.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weightsHiddenOutput, weightsHiddenOutput.data(), weightsHiddenOutput.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, labels[i].data(), outputSize * sizeof(double), cudaMemcpyHostToDevice);

            // Propagación hacia adelante
            forwardKernel << <1, max(hiddenSize, outputSize) >> > (d_input, d_weightsInputHidden, d_hidden, d_weightsHiddenOutput, d_output);
            cudaDeviceSynchronize();
            cudaMemcpy(hidden.data(), d_hidden, hiddenSize * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(output.data(), d_output, outputSize * sizeof(double), cudaMemcpyDeviceToHost);

            // Retropropagación
            backwardKernel << <1, max(hiddenSize, outputSize) >> > (d_input, d_weightsInputHidden, d_hidden, d_weightsHiddenOutput, d_output, d_labels);
            cudaDeviceSynchronize();
            cudaMemcpy(weightsInputHidden.data(), d_weightsInputHidden, weightsInputHidden.size() * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(weightsHiddenOutput.data(), d_weightsHiddenOutput, weightsHiddenOutput.size() * sizeof(double), cudaMemcpyDeviceToHost);

            if (i % 1000 == 0) {
                int errors = testing();
                float accuracy = 1.0 - ((float)errors / (float)nTest);
                cout << "Iteración: " << i/1000 + (epoch*60) << endl;
                cout << "Precisión: " << accuracy * 100 << "%" << endl;
                cout << "Error: " << (float)errors / (float)nTest * 100 << "%" << endl;
                cout << "--------------------" << endl;
                outFileAccuracy << i / 1000 + (epoch * 60) << " " << accuracy * 100 << "\n";
            }
        }
        
    }
    outFileAccuracy.close();
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weightsInputHidden);
    cudaFree(d_weightsHiddenOutput);
    cudaFree(d_labels);
}


int main() {
    srand(time(NULL));

    // Inicializar los pesos
    initializeWeights(weightsInputHidden);
    initializeWeights(weightsHiddenOutput);

    vector<vector<double>> images(nTraining, vector<double>(inputSize, 0.0));
    vector<vector<double>> labels(nTraining, vector<double>(outputSize, 0.0));

    // Cargar datos de entrenamiento
    loadMNISTData(training_image_fn, training_label_fn, images, labels, nTraining);

    // Archivo para guardar la precisión
    std::ofstream outFileAccuracy("accuracy_data.txt");
    if (!outFileAccuracy) {
        cerr << "No se pudo abrir el archivo de precisión para escribir." << endl;
        return 1;
    }

    // Entrenar la red y medir la precisión
    for (int iteration = 1; iteration <= 3; ++iteration) {
        trainNetwork(images, labels);
        int errors = testing();
        float accuracy = 1.0 - ((float)errors / (float)nTest);
        cout << "Iteración: " << iteration*60 +1 << endl;
        cout << "Precisión: " << accuracy * 100 << "%" << endl;
        cout << "Error: " << (float)errors / (float)nTest * 100 << "%" << endl;
        cout << "--------------------" << endl;
        outFileAccuracy << iteration * 60 + 1 << " " << accuracy * 100 << "\n";
    }

    outFileAccuracy.close();

    // Guardar los pesos entrenados
    ofstream outFile("weights.txt");
    if (!outFile) {
        cerr << "No se pudo abrir el archivo para escribir los pesos." << endl;
        return 1;
    }
    for (const double& weight : weightsInputHidden) {
        outFile << weight << "\n";
    }
    for (const double& weight : weightsHiddenOutput) {
        outFile << weight << "\n";
    }
    outFile.close();

    return 0;
}
