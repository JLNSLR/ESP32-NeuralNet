#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

//#define NN_DEBUG


enum nn_activation_f { Linear, ReLu, leakyReLu };

enum nn_loss_function { MSE, MAE, huber_loss };

enum nn_regularization { none, ridge, lasso };

struct sample_t {
    float* inputs;
    float* outputs;
};

struct batch_t {
    sample_t* samples;
    int n_samples;
};


struct diff_fct_out {
    float x;
    float derivative;
};

struct sample_loss_t {
    float* input;
    diff_fct_out* loss;
};

struct batch_loss_t {
    sample_loss_t* samples;
    int n_samples;
};


struct nn_model_weights {
    int n_weights;
    float* weights;
};

enum grad_descent_update_rule { naive, adam };

class NeuralNetwork {
public:

    NeuralNetwork(int depth, int width[], nn_activation_f f_act_per_layer[]);
    ~NeuralNetwork();

    void propagate_forward();

    float train_SGD(float inputs[], float targets[]);
    float train_SGD(sample_t samples[], int n_samples);
    float train_SGD_ext_loss(float inputs[], float ext_loss[], float ext_loss_derivative[]);

    float train_batch(batch_t batch);

    float train_mini_batches(batch_t batches[], int n_batches);
    float train_mini_batches_ext_loss(batch_loss_t batches[], int n_batches);

    float backpropagation(float inputs[], float targets[]);
    float backpropagation(float inputs[], float loss, float loss_derivatives[]);

    float* predict(float input_vector[]);
    void set_input(float input_vector[]);
    float* get_output();

    float* get_network_derivative(float* input_vector);
    float* get_error_filtered();

    nn_model_weights get_model_weights();
    void load_model_weights(nn_model_weights weights_data);

    void set_output_barriers(bool limit_max[], bool limit_min[], float outputs_min[], float outputs_max[]);

    // Model Parameters
    const int depth;
    const int* width; //array of size depth
    nn_activation_f const* activation_function_per_layer; // array of size depth-1
    float*** weights;


    // Hyperparameters
    float learning_rate = 0.01;
    nn_loss_function loss_type = MSE;
    grad_descent_update_rule update_rule = adam;

    nn_regularization regularization = none;
    float reg_penalty_factor = 1e-8;
    float barrier_softness = 1e-3;

    float max_gradient_abs = 100;


private:

    // Internal values
    float** neuron_outputs;

    // Internal Training Values
    float** neuron_loss;
    float** preact;
    float*** weights_grad; //gradient 
    int n_weights;

    float current_error;
    float prev_error;


    // Core Functions for Training
    float apply_activation_function(float x, int layer);
    float grad_activation_function(float x, int layer);
    void apply_gradient_descent(grad_descent_update_rule update_method = adam);
    diff_fct_out get_loss(float x, nn_loss_function loss_type);
    float clip_gradient(float input_grad);


    // Activation Functions
    float f_ReLu(float x);
    float grad_ReLu(float x);
    float f_leakyReLu(float x);
    float grad_leakyReLu(float x);


    // Gradient Descent Method Variables
    const float beta_1 = 0.9;
    const float beta_2 = 0.999;
    float*** adam_m;
    float*** adam_v;

    // Output Barrier Parameters
    bool output_barriers = false;
    bool* output_limits_max;
    bool* output_limits_min;
    float* outputs_max;
    float* outputs_min;
    diff_fct_out* barrier_loss;


    //Helper Functions
    diff_fct_out* get_barrier_loss();
    float fast_inv_sqrt(float number);
    float random_float(float max, float min);

};




#endif // !NEURAL_NETWORK_H
