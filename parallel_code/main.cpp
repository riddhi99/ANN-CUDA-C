#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <random>
#include "cublas_v2.h"
#include "Neural_Network.h"
#include <vector>
#include "helper.h"
#include <sys/time.h>
#include <time.h>

struct timeval start, stop;

int main()
{

		FILE* fp;
		fp = fopen("block_parallel_32.txt", "w");

		int block_size;

		for(block_size=256; block_size<=1024; block_size*=2)
		{
				int training_sample_size = 50000;
				float epsilon = 2.0;
				int epochs = 1000;
				int image_dimension = 784;
  			std::vector<int> neurons_count;
  			neurons_count.push_back(image_dimension);
  			neurons_count.push_back(128);
  			neurons_count.push_back(10);

  			std::ifstream image_file;
  			int learn_b_s = 10000;
  			Neural_Network training_neural_network(neurons_count.size(), neurons_count, learn_b_s);

  			image_file.open("./data/train_image.txt", std::ifstream::in);
  			float* learn_sample = new float[training_sample_size*image_dimension];
  			for(int i = 0; i < training_sample_size; i++)
  			{
    				for(int j = 0; j < image_dimension; j++)
      					image_file >> learn_sample[INDEX(i,j,training_sample_size)];
  			}
  			image_file.close();

  			int *learn_l;
  			learn_l = new int[training_sample_size];
  			image_file.open("./data/train_label.txt", std::ifstream::in);
  			for(int i = 0; i < training_sample_size; i++)
      			image_file >> learn_l[i];
  			image_file.close();


  			int val_sample_s = 10000;
  			Neural_Network val_neural_network(neurons_count.size(), neurons_count, val_sample_s);


  			image_file.open("./data/validation_image.txt", std::ifstream::in);
  			float* val_img = new float[val_sample_s*image_dimension];
  			for(int k = 0; k < val_sample_s; k++)
  			{
    				for(int m = 0; m < image_dimension; m++)
      					image_file >> val_img[INDEX(k,m,val_sample_s)];
  			}
  			image_file.close();

  			int *val_l;
  			val_l = new int[val_sample_s];
  			image_file.open("./data/validation_label.txt", std::ifstream::in);
  			for(int k = 0; k < val_sample_s; k++)
    				image_file >> val_l[k];
  			image_file.close();
  			val_neural_network.batch_setter(val_img, val_l, val_sample_s, image_dimension);

  			float* learn_img_b = new float [learn_b_s*image_dimension];
  			int* learn_l_b = new int [learn_b_s];

  			std::vector<int> index_s;
  			for(int p = 0; p < training_sample_size; p++)
    				index_s.push_back(p);

				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start);

  			for(int n = 0; n < epochs; n++)
  			{
    				std::random_shuffle(index_s.begin(), index_s.end());
    				for(int i = 0; i < learn_b_s; i++)
    				{
      					for(int j = 0; j < image_dimension; j++)
										learn_img_b[INDEX(i,j,learn_b_s)] = learn_sample[INDEX(index_s[i],j,training_sample_size)];

      					learn_l_b[i] = learn_l[index_s[i]];
    				}

    				training_neural_network.batch_setter(learn_img_b, learn_l_b, learn_b_s, image_dimension);
    				float loss = training_neural_network.learning(1, epsilon, block_size);
    				val_neural_network.wt_setter(training_neural_network.wt_getter());
    				val_neural_network.bias_setter(training_neural_network.bias_getter());
    				val_neural_network.forward_prop(block_size);

  			}
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);

				float milliseconds = 0;
				cudaEventElapsedTime(&milliseconds, start, stop);

  			int testing_sample_s = 10000;
  			image_file.open("./data/test_image.txt", std::ifstream::in);
  			float* testing_img = new float[testing_sample_s*image_dimension];
  			for(int i = 0; i < testing_sample_s; i++)
  			{
    				for(int j = 0; j < image_dimension; j++)
      					image_file >> testing_img[INDEX(i,j,testing_sample_s)];
  			}
  			image_file.close();

  			int *org_l;
  			org_l = new int[testing_sample_s];
  			image_file.open("./data/test_label.txt", std::ifstream::in);
  			for(int i = 0; i < testing_sample_s; i++)
  			{
    				image_file >> org_l[i];
  			}
  			image_file.close();

  			Neural_Network testing_neural_network(neurons_count.size(), neurons_count, testing_sample_s);
  			testing_neural_network.batch_setter(testing_img, org_l, testing_sample_s, image_dimension);
  			testing_neural_network.wt_setter(training_neural_network.wt_getter());
  			testing_neural_network.bias_setter(training_neural_network.bias_getter());
  			testing_neural_network.forward_prop(block_size);

				fprintf(fp, "%d %lf", block_size, milliseconds);
		}
		fclose(fp);
		return 0;
}
