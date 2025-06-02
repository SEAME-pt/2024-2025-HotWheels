/**
 * @file yolo_infer.cpp
 * @brief Inferência de objetos usando YOLOv5 com TensorRT e OpenCV CUDA.
 *
 * Este programa realiza inferência de detecção de objetos em tempo real utilizando um modelo YOLOv5 otimizado com TensorRT,
 * acelerado por CUDA, e exibe os resultados com OpenCV. Suporta leitura de labels, pós-processamento com NMS e exibição de FPS.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <chrono>

using namespace nvinfer1;
using namespace std;

#define CLASSES 91 ///< Número de classes do modelo YOLOv5
#define MODEL "models/yolov5m.engine" ///< Caminho do engine TensorRT

/**
 * @class LabelManager
 * @brief Gerencia o carregamento e acesso às labels das classes.
 */
class LabelManager {
private:
	std::vector<std::string> labels; ///< Lista de labels

public:
	/**
	 * @brief Construtor que carrega as labels de um arquivo.
	 * @param labelPath Caminho para o arquivo de labels.
	 */
	LabelManager(const std::string& labelPath) {
		loadLabels(labelPath);
	}

	/**
	 * @brief Carrega as labels do arquivo especificado.
	 * @param labelPath Caminho para o arquivo de labels.
	 */
	void loadLabels(const std::string& labelPath) {
		std::ifstream file(labelPath);
		if (!file.is_open()) {
			cerr << "[ERRO] Não foi possível abrir o arquivo de labels: " << labelPath << endl;
			return;
		}

		std::string line;
		while (std::getline(file, line)) {
			line.erase(0, line.find_first_not_of(" \t\r\n"));
			line.erase(line.find_last_not_of(" \t\r\n") + 1);
			labels.push_back(line);
		}
		file.close();

		cout << "[INFO] Carregadas " << labels.size() << " labels." << endl;
	}

	/**
	 * @brief Retorna o nome da label para um dado classId.
	 * @param classId Índice da classe.
	 * @return Nome da classe ou "Unknown".
	 */
	std::string getLabel(int classId) const {
		// Correção: cast para evitar warning signed/unsigned
		if (classId >= 0 && static_cast<size_t>(classId) < labels.size()) {
			return labels[classId];
		}
		return "Unknown";
	}

	/**
	 * @brief Retorna o número de classes carregadas.
	 * @return Número de classes.
	 */
	size_t getNumClasses() const {
		return labels.size();
	}
};

/**
 * @brief Calcula o volume (número total de elementos) de um tensor dado suas dimensões.
 * @param dims Dimensões do tensor.
 * @return Volume total.
 */
size_t calculateVolume(const nvinfer1::Dims& dims) {
	size_t volume = 1;
	for (int i = 0; i < dims.nbDims; ++i) {
		volume *= dims.d[i];
	}
	return volume;
}

/**
 * @brief Inicializa a câmera usando pipeline GStreamer.
 * @return Objeto cv::VideoCapture aberto.
 */
cv::VideoCapture initCamera() {
	std::string pipeline =
		"nvarguscamerasrc sensor_mode=4 ! "
		"video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
		"nvvidconv flip-method=0 ! "
		"video/x-raw,format=BGRx ! "
		"videoconvert ! "
		"video/x-raw,format=BGR ! "
		"appsink drop=1";

	cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
	if (!cap.isOpened()) {
		cerr << "[ERRO] Falha ao abrir a câmera!" << endl;
		exit(EXIT_FAILURE);
	}
	return cap;
}
