#ifndef _NEURAL_NETWORK_H
#define _NEURAL_NETWORK_H 1

#include "cublas_v2.h"
#include <vector>

class Neural_Network
{
  cublasHandle_t handle;
  float learning_rate = 1.0;
  /* size of batch = number images that will be processed together */
  int size_batch;
  /* Labels of image on cpu pointer */
  int *cpu_label;
  /* Weights and bias pointers on CPU */
  float **cpu_wt, **cpu_bias;
  /* Number of Layers in Neural Network */
  int layer_count;
  float *gpu_1;
  /* Labels of image on gpu pointer */
  int *gpu_label;
  /* Weights and bias pointers on CPU */
  float **gpu_wt, **gpu_bias;
  /* Activation of neurons in GPU */
  float **gpu_activation;
  /* Probablility for the output layer using soft_max_func */
  float *cpu_prob,*gpu_prob;
  /* Number of Neurons in each layer of Neural Network */
  std::vector<int> neurons_count;
  /* Derivative wrt weights */
  float **gpu_dwt;
  /* Derivative wrt bias */
  float **gpu_dbias;
  /* Predicted output/expected_op for the current training batch sample */
  int *cpu_label_predicted;
  /* Derivative wrt input neurons */
  float **gpu_delta;
  float b = 0.0;

 public:
  Neural_Network(int layers, std::vector<int> nodes, int size);
  int get_layers_count();
  void forward_prop(int blocksize);
  void bias_setter(float **bias_in);
  void wt_setter(float **weights_in);
  float** wt_getter();
  float** bias_getter();
  float accuracy_calculation();
  void backward_prop(int blocksize);
  float learning(int epochs, float eta, int blocksize);
  float cost_calculation();
  void batch_setter(float *data, int *expected_op, int size, int image_dimension);
};

#endif
