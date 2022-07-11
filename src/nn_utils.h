#ifndef NN_UTILS_H
#define NN_UTILS_H

#include<Arduino.h>
#include <NeuralNetwork.h>

void nn_save_model_weights_on_flash(nn_model_weights weight_data, char model_name[]);

nn_model_weights nn_load_model_weights_from_flash(char model_name[], int n_weights);

void nn_clear_data_on_flash(char model_name[]);



#endif // !NN_UTILS_H