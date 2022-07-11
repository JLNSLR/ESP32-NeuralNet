#include <Arduino.h>

#include<NeuralNetwork.h>

const int depth = 6;

int width[depth] = { 2,8,8,6,6,1 };

nn_activation_f activation[depth - 1] = { leakyReLu, leakyReLu, leakyReLu, Linear,Linear };

NeuralNetwork* net;


float delta_t = 1e-3;



void setup() {
  Serial.begin(115200);
  Serial.println("Neural Network test");

  net = new NeuralNetwork(depth, width, activation);

  net->update_rule = adam;

  net->learning_rate = 0.001;

  net->regularization = none;

}

void loop() {

  static int long n = 0;

  float x = 1.0 * sin(delta_t * n * 2 * PI);
  float x_2 = 3.0 * sin(delta_t * n * 1.0 * PI);

  long time_1 = micros();
  float* y_hat = net->predict(&x);
  long time_2 = micros();

  //Serial.print("delta propforward: ");
  //Serial.println(time_2 - time_1);

  float input_torque = 1 * (x * x * 0.0 + sin(2 * PI * (x - x_2) * 3) * x + x * 0.2) * x_2 * x_2 - x_2;

  float input_torque_copy = input_torque;
  float x_1 = x;
  float x_2c = x_2;
  float input_copy[2] = { x_1,x_2c };
  n++;

  time_1 = micros();
  float error = net->train_SGD(input_copy, &input_torque_copy);
  time_2 = micros();

  //Serial.print("delta backprop: ");
  //Serial.println(time_2 - time_1);

  delay(1);

  if (n % 10 == 0) {
    Serial.print(input_torque);
    Serial.print("\t");
    Serial.print(*y_hat);
    Serial.print("\t");
    Serial.println(error);

  }


}