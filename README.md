# ESP32-NeuralNet
WIP. Lightweight Implementation of a versatile Feed-Forward Neural Network that can be trained online on ESP32 Microcontrollers. 

- Online Training via Backpropagation
- Stochastic Gradient Descent for Online Training
  - Naive Gradient Descent
  - Adaptive Learning Rate with ADAM implemented
- Mini-Batch Gradient Descent 
- Regularization Methods to reduce overfitting and keep weights small
  - L2 (Ridge Regression) 
  - L2 (Lasso)
- Activation Functions Implemented:
  - ReLu 
  - leaky ReLu 
  - Linear
- Implemented Barrier Methods for Constrained Outputs 
- focused on computationally efficient methods 
  - Gradient Clipping to maintain numerical stability on Single Precision Float System
- Use Arduino-ESP32 Preferences Library to save weights on Flash

