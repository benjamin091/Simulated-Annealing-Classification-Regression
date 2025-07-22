# Simulated Annealing for Neural Network Configuration  
Metaheuristic Optimization for Classification and Regression

## Project Overview

This project implements a framework for automatic configuration of neural networks (NN) applied to classification and regression tasks. The goal is to efficiently search the large space of possible NN architectures, including variations in activation functions, number of layers, and number of nodes per layer.

The core optimization method is **Simulated Annealing**, a probabilistic local search technique that iteratively improves NN configurations by exploring the search space and avoiding local minima through controlled acceptance of worse solutions.

## Key Features

- Implementation of Simulated Annealing as a metaheuristic optimization method for NN architecture tuning.  
- Support for different activation functions, layer counts, and nodes per layer.  
- Evaluation based on cross-validation performance metrics for classification and regression.  
- Use of existing implementations or custom code for forward and backward propagation.  
- Fully implemented search algorithm from scratch, without external libraries for optimization.

## Methodology

- **Simulated Annealing**: Probabilistic local search that explores configurations by accepting improvements and, with decreasing probability, worse solutions to escape local optima.  
- Performance evaluation at each step uses appropriate metrics for classification (e.g., accuracy, F1-score) and regression (e.g., RMSE).  

## Usage

- Define the search space with ranges for activation functions, number of layers, and nodes per layer.  
- Run simulated annealing to iteratively optimize the NN architecture.  
- Analyze results to identify the best-performing configurations.

## Tools & Technologies

- Python  
- Custom implementation of simulated annealing optimizer  
- Existing or custom NN training code (forward/backward propagation)  
- Cross-validation and performance metric functions

## Deliverables

- Source code for simulated annealing optimizer  
- Neural network training and evaluation scripts  
- Documentation and analysis of experimental results  

---

## Contact

For questions or feedback, please open an issue or contact the maintainer.

