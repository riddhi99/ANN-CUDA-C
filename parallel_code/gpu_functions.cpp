#include "gpu_functions.h"
#include "helper.h"

/* Kernel function to increment the epoch level for all the batch images */
__global__ void epoch_incrementer(float *weight, float *dweight, int n, float eta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    weight[idx] = weight[idx] + eta * dweight[idx];
}

/* Gradient descent of softmax */
__global__ void grad_softmax_calculation(int *expected_op, float* prob, int m, int n, float* grad)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    for (int i = 0; i < n; i++)
    {
      if (expected_op[idx] == i )
	     grad[INDEX(idx,i,m)] = 1 - prob[INDEX(idx,i,m)];
      else
	     grad[INDEX(idx,i,m)] = - prob[INDEX(idx,i,m)];
    }
  }
}

/* Sigmoid function to convert into probabiliy */
__global__ void sigmoid_func(float *activation, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    activation[idx] = 1 / (1 + expf(-activation[idx]));
};

/* Gradient calculation for back propagation in neural network */
__global__ void gradient_calculation(float *grad, float* activation, int m, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m*n)
    grad[idx] = grad[idx] * activation[idx] * (1-activation[idx]);
}

/* softmax function used for getting the cummulative probability for the batch images */
__global__ void soft_max_func(float *activation, float* prob, int m, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    float max = activation[INDEX(idx,0,m)];
    for ( int i = 1; i < n; i++)
    {
      if (activation[INDEX(idx,i,m)] > max)
	       max = activation[INDEX(idx,i,m)];
    }
    float s = 0;
    for(int i = 0; i < n; i++)
    {
      prob[INDEX(idx,i,m)] = expf(activation[INDEX(idx,i,m)] - max);
      s = s + prob[INDEX(idx,i,m)];
    }

    for(int i = 0; i < n; i++)
      prob[INDEX(idx,i,m)] = prob[INDEX(idx,i,m)] / s;
  }
}

/* function to calculate the exponential of the activation matrix of a layer*/
__global__ void exponential_matrix(float *activation, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    activation[idx] = expf(activation[idx]);
};
