import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna

# Define the Autoencoder architecture 
def Autoencoder(trial, input_size, bottleneck_neurons, n_min, 
                n_max, min_layers, max_layers):
    # define the lists containing the encoder and decoder layers
    encoder_layers = []
    decoder_layers = []

    # define a container for out_features
    out_features   = []
    
    n_layers = trial.suggest_int("n_layers", min_layers, max_layers)
    for i in range(n_layers):
        if i == 0: 
            if i == n_layers - 1:  # if only 1 hidden layer
                # Add encoder input layer 
                encoder_layers.append(nn.Linear(input_size, bottleneck_features))
                encoder_layers.append(nn.LeakyReLU(0.2))

                # Add final decoder output layer
                decoder_layers.append(nn.Linear(bottleneck_features, input_size))
                # No activation layer here (decoder output)
                
            else: 
            # Define out_features
            out_features.append(trial.suggest_int("n_units_{}".format(i), n_min, n_max))
        
            # Add encoder input layer 
            encoder_layers.append(nn.Linear(input_size, out_features[0]))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
            # Define in_features to be out_features for decoder
            in_features = out_features[0]
        
            # Add final decoder output layer
            decoder_layers.append(nn.Linear(in_features, input_size))
            # No activation layer here (decoder output)
    
        elif i == n_layers - 1:
            # add the layers adjacent to the bottleneck
            encoder_layers.append(nn.Linear(in_features, bottleneck_neurons))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
            decoder_layers.append(nn.Linear(bottleneck_neurons, in_features))
            decoder_layers.append(nn.LeakyReLU(0.2))
        
        else:
            # Define out_features
            out_features.append(trial.suggest_int("n_units_{}".format(i), n_min, n_max))
        
            # Add encoder layers
            encoder_layers.append(nn.Linear(in_features, out_features[i]))
            encoder_layers.append(nn.LeakyReLU(0.2))
        
            # Define in_features to be out_features for decoder
            in_features = out_features[i] 

            # Add decoder layers
            decoder_layers.append(nn.Linear(in_features, out_features[i-1]))
            decoder_layers.append(nn.LeakyReLU(0.2))

    # Reverse order of layers in decoder list
    decoder_layers.reverse()

    # Complete layers list (symmetric)
    layers = encoder_layers + decoder_layers

    # return the model
    return nn.Sequential(*layers)
