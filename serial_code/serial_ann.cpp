/* This is the serial code for training the neural network for 60000 MNIST
 * dataset images.
 * IDs: 201601124, 201601082, 201601210, 201601077
 */

#include <iostream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <iterator>
#include <cmath>
#include <algorithm>
#include <set>
#include <sys/time.h>
#include <time.h>
#include <fstream>

struct timeval start, stop;

using namespace std;

/* Initializing with expected_op image filename */
const string training_label_fn = "mnist/learning-labels.idx1-ubyte";

/* Initializing with training image filename */
const string training_image_fn = "mnist/learning-images.idx3-ubyte";

/* Number of training samples */
const int training_samples = 60000;

/* Image size in MNIST dataset */
const int imageHeight = 28, imageWidth = 28;

/* image matrix, the size of the image is 28x28 MNIST gray scale image*/
int d[imageWidth + 1][imageHeight + 1];

/* Number of neurons in input layer, here input_neurons = 784*/
const int input_neurons = imageWidth * imageHeight;

/* Number of neurons in hidden layer */
const int hidden_neurons = 64;

/* Number of neurons in output layer, here output_neurons = 10; 0-9 classes */
const int output_neurons = 10;

/* Threshold to determine whether the required value has reached its optimal value, basically maximum error allowed*/
const double eps = 1e-3;

/* Number of iterations required for back propagation */
const int Epochs = 512;

/* Parameter to fasten (Momentum) the process to reach optimal value during back propagation*/
const double mv = 0.9;

/* Learning Rate */
const double learning_rate = 1e-3;

/* weight and gradient between layer 1 (input layer) & 2 (hidden layer) */
double *wt_layer12[input_neurons + 1], *wt_gradient12[input_neurons + 1], *output1;

/* weight and gradient between layer 2 (hidden layer) & 3 (output layer) */
double *wt_layer23[hidden_neurons + 1], *wt_gradient23[hidden_neurons + 1], *input2, *output2, *theta2;

/* Layer 3 (Output layer) */
double *input3, *output3, *theta3;
double expected[output_neurons + 1];

/* File pointer for reading the sample image file and expected_op file */
ifstream sample_file, label_file;

/* This function reads the sample images and their corresponding from the files mentioned */
void take_input() {

    char number;
    for (int i = 1; i <= imageWidth; i++)
		{
    		for (int j = 1; j <= imageHeight; j++)
				{
      			sample_file.read(&number, sizeof(char));
        		if(number == 0)
						{
								d[i][j] = 0;
						}
						else
						{
								d[i][j] = 1;
						}
        }
		}

    for (int i = 1; i <= imageWidth; i++)
		{
        for (int j = 1; j <= imageHeight; j++)
				{
            int pos = j + (i - 1) * imageHeight;
            output1[pos] = d[i][j];
        }
		}

    label_file.read(&number, sizeof(char));
    for (int i = 1; i <= output_neurons; i++)
		{
				expected[i] = 0.0;
		}
    expected[number + 1] = 1.0;
}

/* Initialize the arrays*/
void initialize()
{
		/* Declaration of the 2D matrix of weights and gradient for input layer (layer 1) and hidden layer (layer 2) */
    for(int i = 1; i <= input_neurons; i++)
		{
        wt_layer12[i] = new double [hidden_neurons + 1];
        wt_gradient12[i] = new double [hidden_neurons + 1];
    }

    output1 = new double [input_neurons + 1];

		/* Declaration of the 2D matrix of weights and gradient for hidden layer (layer 2) and output layer (layer 3) */
    for (int i = 1; i <= hidden_neurons; i++)
		{
        wt_layer23[i] = new double [output_neurons + 1];
        wt_gradient23[i] = new double [output_neurons + 1];
    }

		/* Hidden Layer */
    input2 = new double [hidden_neurons + 1];
    output2 = new double [hidden_neurons + 1];
    theta2 = new double [hidden_neurons + 1];

		/* Output Layer */
    input3 = new double [output_neurons + 1];
    output3 = new double [output_neurons + 1];
    theta3 = new double [output_neurons + 1];

    /* Initializing weights from Layer 1 to Layer 2 */
    for (int i = 1; i <= input_neurons; i++)
		{
        for (int j = 1; j <= hidden_neurons; j++)
				{
            int sign = rand() % 2;
            wt_layer12[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1)
						{
								wt_layer12[i][j] = - wt_layer12[i][j];
						}
        }
		}
		/* Initializing weights from Layer 2 to Layer 3 */
    for (int i = 1; i <= hidden_neurons; i++)
		{
        for (int j = 1; j <= output_neurons; j++)
				{
            int temp = rand()%2;
            wt_layer23[i][j] = (double)(rand()% 10+1)/(10.0*output_neurons);
            if (temp == 1)
						{
								wt_layer23[i][j] = - wt_layer23[i][j];
						}
        }
		}
}

/* This is the loss function used to calculate the error of the output layer.
 * This is the first function used while computing the back propagation.
 */
double error()
{
    double ans = 0.0;
    for (int i = 1; i <= output_neurons; i++)
		{
        ans += (output3[i] - expected[i]) * (output3[i] - expected[i]);
		}
    ans *= 0.5;
    return ans;
}

/* Sigmoid Function to convert the number into probability 0-1 number */
double sigmoid_func(double a)
{
    return 1.0 / (1.0 + exp(-a));
}

/* Back Propagation */
void backward_prop()
{
    double sum;

    for (int i = 1; i <= output_neurons; i++)
		{
        theta3[i] = output3[i] * (1 - output3[i]) * (expected[i] - output3[i]);
		}

    for (int i = 1; i <= hidden_neurons; i++)
		{
        sum = 0.0;
        for (int j = 1; j <= output_neurons; j++)
				{
            sum += wt_layer23[i][j] * theta3[j];
				}
        theta2[i] = output2[i] * (1 - output2[i]) * sum;
    }

    for (int i = 1; i <= hidden_neurons; i++)
		{
        for (int j = 1; j <= output_neurons; j++)
				{
            wt_gradient23[i][j] = (learning_rate * theta3[j] * output2[i]) + (mv * wt_gradient23[i][j]);
            wt_layer23[i][j] += wt_gradient23[i][j];
        }
		}

    for (int i = 1; i <= input_neurons; i++)
		{
        for (int j = 1 ; j <= hidden_neurons ; j++)
				{
            wt_gradient12[i][j] = (learning_rate * theta2[j] * output1[i]) + (mv * wt_gradient12[i][j]);
            wt_layer12[i][j] += wt_gradient12[i][j];
        }
		}
}

/* Forward Propagation */
void forward_prop()
{
    for (int i = 1; i <= hidden_neurons; i++)
		{
				input2[i] = 0.0;
		}

    for (int i = 1; i <= output_neurons; i++)
		{
				input3[i] = 0.0;
		}

    for (int i = 1; i <= input_neurons; i++)
		{
        for (int j = 1; j <= hidden_neurons; j++)
				{
            input2[j] += output1[i] * wt_layer12[i][j];
				}
		}

    for (int i = 1; i <= hidden_neurons; i++)
		{
				output2[i] = sigmoid_func(input2[i]);
		}

    for (int i = 1; i <= hidden_neurons; i++)
		{
        for (int j = 1; j <= output_neurons; j++)
				{
            input3[j] += output2[i] * wt_layer23[i][j];
				}
		}

    for (int i = 1; i <= output_neurons; i++)
		{
				output3[i] = sigmoid_func(input3[i]);
		}
}

/* Training Process = Forward Propagation + Back Propagation for each sample image */
void training_process()
{
    for (int i = 1; i <= input_neurons; i++)
		{
        for (int j = 1; j <= hidden_neurons; j++)
				{
						wt_gradient12[i][j] = 0.0;
				}
		}

    for (int i = 1; i <= hidden_neurons; i++)
		{
        for (int j = 1; j <= output_neurons; j++)
				{
						wt_gradient23[i][j] = 0.0;
				}
		}

    for (int i = 1; i <= Epochs; i++)
		{
        forward_prop();
        backward_prop();
        if (error() < eps)
				{
						return;
				}
    }
    return;
}


int main(int argc, char *argv[])
{

			/* Binary File of Labels to be used in training process */
			label_file.open(training_label_fn.c_str(), ios::in | ios::binary );
			/*Binary File of Images to be used in training process */
			sample_file.open(training_image_fn.c_str(), ios::in | ios::binary);

    	char number;
			/* Following loops will read the headers of the file */
			for (int i = 1; i <= 16; i++)
			{
        	sample_file.read(&number, sizeof(char));
			}
    	for (int i = 1; i <= 8; i++)
			{
        	label_file.read(&number, sizeof(char));
			}

			/* Neural Network Initialization */
    	initialize();

			/* Time calculations */
			double end_time;
			gettimeofday(&start, NULL);
			double start_t = start.tv_sec*1000000 + start.tv_usec;

    	for (int sample = 1; sample <= training_samples; sample++)
			{
					/* Read the files for images and respective labels */
        	take_input();
					/* Training Process */
					training_process();
			}

			gettimeofday(&stop,NULL);
			double stop_t = stop.tv_sec*1000000+stop.tv_usec;
			end_time = (stop_t - start_t)/1000;

			cout<<"Time elapsed in milliseconds: "<<end_time<<endl;
    	sample_file.close();
    	label_file.close();
    	return 0;
}
