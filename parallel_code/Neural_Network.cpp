#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "Neural_Network.h"
#include "functions.h"
#include "kernels.h"
#include <random>
#include <math.h>
#include <cuda.h>
#include "cublas_v2.h"

Neural_Network::Neural_Network(int layers, std::vector<int> nodes, int size)
{
  cublasStatus_t stat = cublasCreate( &handle );
  size_batch = size;
  neurons_count = nodes;
  layer_count = layers;

  float *cpu_1;
  cpu_1 = new float[size_batch];

  for(int j = 0; j < size_batch; j++)
  {
    cpu_1[j] = 1;
  }

  cudaMalloc( (void **)&gpu_1, sizeof(float)*size_batch );
  cudaMemcpy(gpu_1, cpu_1, sizeof(float)*size_batch, cudaMemcpyHostToDevice);
  delete[] cpu_1;

  gpu_wt = new float* [layer_count-1];
  gpu_dwt = new float* [layer_count-1];
  cpu_wt = new float* [layer_count-1];

  gpu_dbias = new float* [layer_count-1];
  cpu_bias = new float* [layer_count-1];
  gpu_bias = new float* [layer_count-1];


  for (int i = 0; i < layer_count - 1; i++)
  {
    cpu_wt[i] = new float [neurons_count[i]*neurons_count[i+1]];
    cpu_bias[i] = new float [neurons_count[i+1]];

    cudaMalloc( (void **)&gpu_bias[i],  sizeof(float)*neurons_count[i+1] );
    cudaMalloc( (void **)&gpu_dbias[i],  sizeof(float)*neurons_count[i+1] );
    cudaMalloc( (void **)&gpu_wt[i],  sizeof(float)*neurons_count[i]*neurons_count[i+1] );
    cudaMalloc( (void **)&gpu_dwt[i],  sizeof(float)*neurons_count[i]*neurons_count[i+1] );

  }


  std::uniform_real_distribution<float> uniform(-1.0,1.0);

  for (int k = 0; k < layer_count - 1; k++)
  {
    for(int i = 0; i < neurons_count[k]; i++)
    {
      for(int j = 0; j < neurons_count[k+1]; j++)
      {
	        cpu_wt[k][INDEX(i,j,neurons_count[k])] = uniform(Random_Gen());
      }
    }

    for(int i = 0; i < neurons_count[k+1]; i++)
    {
      cpu_bias[k][i] = 0;
    }
  }


  for (int k = 0; k < layer_count - 1; k++)
  {
    cudaMemcpy( gpu_wt[k], cpu_wt[k], sizeof(float)*neurons_count[k]*neurons_count[k+1], cudaMemcpyHostToDevice);
    cudaMemcpy( gpu_bias[k], cpu_bias[k], sizeof(float)*neurons_count[k+1], cudaMemcpyHostToDevice);
  }

  gpu_activation = new float* [layer_count];
  for (int k = 0; k < layer_count; k++)
  {
    cudaMalloc( (void **)&gpu_activation[k], sizeof(float)*size_batch*neurons_count[k]);
  }

  cpu_prob = new float[size_batch*neurons_count[layer_count-1]];
  cudaMalloc( (void **)&gpu_prob, sizeof(float)*size_batch*neurons_count[layer_count-1] );

  cpu_label_predicted = new int [size_batch];

  cpu_label = new int [size_batch];
  cudaMalloc( (void **)&gpu_label, sizeof(int)*size_batch );

  gpu_delta = new float* [layer_count-1];
  for (int l = 0; l < layer_count - 1; l++)
  {
    cudaMalloc( (void **)&gpu_delta[l], sizeof(float)*size_batch*neurons_count[l+1] );
  }
};

void Neural_Network::bias_setter(float **bias_in)
{
  for (int k = 0; k < layer_count - 1; k++)
  {
    cudaMemcpy(gpu_bias[k], bias_in[k], sizeof(float)*neurons_count[k+1], cudaMemcpyHostToDevice);
  }
}


int Neural_Network::get_layers_count()
{
  return layer_count;
}


void Neural_Network::batch_setter(float *data, int *expected_op, int size, int image_dimension)
{
  assert(neurons_count[0] == image_dimension);
  assert(size_batch == size);
  cudaMemcpy(gpu_label, expected_op, sizeof(int)*size_batch, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_activation[0], data, sizeof(float)*size_batch*neurons_count[0], cudaMemcpyHostToDevice);

  for(int p = 0; p < size_batch; p++)
  {
    cpu_label[p] = expected_op[p];
  }
}

void Neural_Network::forward_prop(int blocksize)
{
  learning_rate = 1.0;
  for (int k = 1; k < layer_count; k++)
  {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
    		 size_batch, neurons_count[k] , neurons_count[k-1],
    		 &learning_rate,
    		 gpu_activation[k-1], size_batch,
    		 gpu_wt[k-1], neurons_count[k-1],
    		 &b,
    		 gpu_activation[k], size_batch);
    cublasSger( handle, size_batch, neurons_count[k], &learning_rate, gpu_1, 1, gpu_bias[k-1], 1, gpu_activation[k], size_batch);
    if ( k < layer_count - 1)
    {
      sigmoid_func<<< size_batch*neurons_count[k]/blocksize + 1 , blocksize>>> (gpu_activation[k], size_batch*neurons_count[k]);
    }
  }

  soft_max_func<<< size_batch/blocksize+1, blocksize >>> (gpu_activation[layer_count-1], gpu_prob, size_batch, neurons_count[layer_count-1]);
  cudaMemcpy(cpu_prob, gpu_prob, sizeof(float)*size_batch*neurons_count[layer_count-1], cudaMemcpyDeviceToHost);
}


float Neural_Network::cost_calculation()
{
  float loss = 0;
  for(int i = 0; i < size_batch; i++)
  {
    loss += -log(cpu_prob[INDEX(i,cpu_label[i],size_batch)]);
  }
  return loss/size_batch;
}


float Neural_Network::learning(int epochs, float eta, int blocksize)
{
  for (int n = 0; n < epochs; n++)
  {
    forward_prop(blocksize);
    backward_prop(blocksize);
    for (int l = 0; l < layer_count - 1; l++)
    {
      epoch_incrementer<<< neurons_count[l]*neurons_count[l+1]/blocksize + 1, blocksize >>> (gpu_wt[l], gpu_dwt[l], neurons_count[l]*neurons_count[l+1], eta);
      epoch_incrementer<<<neurons_count[l+1]/blocksize + 1, blocksize >>> (gpu_bias[l], gpu_dbias[l], neurons_count[l+1], eta);
    }
  }
  return cost_calculation();
}


float** Neural_Network::bias_getter()
{
  for (int l = 0; l < layer_count - 1; l++)
  {
    cudaMemcpy(cpu_bias[l], gpu_bias[l], sizeof(float)*neurons_count[l+1], cudaMemcpyDeviceToHost);
  }
  return cpu_bias;
}

void Neural_Network::backward_prop(int blocksize)
{

  grad_softmax_calculation <<< size_batch/blocksize+1, blocksize >>> (gpu_label, gpu_prob, size_batch, neurons_count[layer_count-1], gpu_delta[layer_count-2]);

  for(int l = layer_count - 3; l >= 0; l--)
  {
    cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_T,
  		 size_batch, neurons_count[l+1] ,neurons_count[l+2],
  		 &learning_rate,
  		 gpu_delta[l+1], size_batch,
  		 gpu_wt[l+1], neurons_count[l+1],
  		 &b,
  		 gpu_delta[l], size_batch
  		 );
    gradient_calculation<<<size_batch*neurons_count[l+1]/blocksize, blocksize>>>(gpu_delta[l], gpu_activation[l+1], size_batch, neurons_count[l+1]);
  }

  learning_rate = 1.0 / size_batch;
  for(int l = 0; l < layer_count - 1; l++)
  {
    cublasSgemm( handle, CUBLAS_OP_T, CUBLAS_OP_N,
		 neurons_count[l], neurons_count[l+1], size_batch,
		 &learning_rate,
		 gpu_activation[l], size_batch,
		 gpu_delta[l], size_batch,
		 &b,
		 gpu_dwt[l], neurons_count[l]
		 );
    cublasSgemv( handle, CUBLAS_OP_T,
    		 size_batch, neurons_count[l+1],
    		 &learning_rate,
    		 gpu_delta[l], size_batch,
    		 gpu_1, 1,
    		 &b,
    		 gpu_dbias[l],1
		 );
  }
};


float Neural_Network::accuracy_calculation()
{

  int  correct_output = 0;
  for(int i = 0; i < size_batch; i++)
  {
    float max = 0;
    cpu_label_predicted[i] = 0;
    for(int j = 0; j < neurons_count[layer_count-1]; j++)
    {
      if (cpu_prob[INDEX(i,j,size_batch)] > max)
      {
	       cpu_label_predicted[i] = j;
	       max = cpu_prob[INDEX(i,j,size_batch)];
      }
    }
    if (cpu_label_predicted[i] == cpu_label[i])
    {
      correct_output++;
    }
  }
  return float(correct_output)/size_batch;
}


float** Neural_Network::wt_getter()
{
  for (int k = 0; k < layer_count - 1; k++)
  {
    cudaMemcpy(cpu_wt[k], gpu_wt[k], sizeof(float)*neurons_count[k]*neurons_count[k+1], cudaMemcpyDeviceToHost);
  }
  return cpu_wt;
}


void Neural_Network::wt_setter(float **weights_in)
{
  for (int k = 0; k < layer_count - 1; k++)
  {
    cudaMemcpy(gpu_wt[k], weights_in[k], sizeof(float)*neurons_count[k]*neurons_count[k+1], cudaMemcpyHostToDevice);
  }
}
