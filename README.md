# Hybrid-PcNN-model
This repository contains the source code and relevant materials for the paper:

**"A hybrid model with a physics-constrained neural network to improve hydrodynamic prediction"**

This work proposes a hybrid deep learning model combining data-driven and physics-based methods to enhance real-time hydrodynamic predictions. A novel **Physics-Constrained Neural Network (PcNN)** is developed, which integrates physical knowledge into a long short-term memory (LSTM) framework through the input structure and custom loss function. The model is validated on the **Middle Route of the South-to-North Water Diversion Project** in China.

---

## Model Description

- **Model Type**: Hybrid model combining LSTM with physical constraints
- **Physics Integration**: 
  - Physical relationships derived from a 1D hydrodynamic model
  - Incorporated into both model inputs and loss functions
- **Target Task**: Predicting real-time offtake discharges and improving water level forecasts

---

## Experimental Environment

- **Platform**: Windows 10
- **Software**: MATLAB R2019b
- **CPU**: Intel Core i7-9700 @ 3.00 GHz  
- **RAM**: 16 GB
