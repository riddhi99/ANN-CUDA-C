#ifndef _GPU_FUNC_H
#define _GPU_FUNC_H 1

__global__ void soft_max_func(float *activation, float* prob, int m, int n);
__global__ void gradient_calculation(float *grad, float* activation, int m, int n);
__global__ void sigmoid_func(float *activation, int n);
__global__ void grad_softmax_calculation(int *expected_op, float* prob, int m, int n, float *grad);
__global__ void epoch_incrementer(float *weight, float *dweight, int n, float eta);
__global__ void exponential_matrix(float *activation, int n);

#endif
