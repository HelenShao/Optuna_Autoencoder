# Optuna Autoencoder
Using Optuna to optimize hyperparameters for autoencoder model implemented in Pytorch

Goal of Project:
To determine how to accruately represent dark matter halos with reduced number of properties using autoencoder and PCA
Finding hidden relationships between the 11 halo properties with symbolic regression using PYSR

Important Features:
1. Dynamic Autoencoder Architecture - located in architecture.py to allow for optuna optimization of model
2. Data - Normalized and split into train, valid, and test loaders for training
3. Training Parameters - how to implement optuna and what parameters are needed (input_size, batch_size, seed, etc)

