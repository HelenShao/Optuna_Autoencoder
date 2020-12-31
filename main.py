import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import optuna

############################ USE GPUS ###############################

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
print('GPU: %s     ||    Training on %s\n'%(GPU,device))


########################## CREATE DATALOADERS ###########################

#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, n_halos, halo_data, batch_size)

train_loader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_Dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_Dataset,  batch_size=batch_size, shuffle=True)


################################# Objective Function #############################

# Create loss and optimizer function for training  
criterion = nn.MSELoss()  

def objective(trial):

    # Generate the model.
    model = architecture.Autoencoder(trial, input_size, bottleneck_neurons, n_min, n_max).to(device)

    # Generate the optimizers, learning_rate, and weight_decay.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    wd = trial.suggest_float("wd", 1e-5, 1e3, log=True)
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr, weight_decay = wd)
    
    # Train the model.
    for epoch in range(num_epochs):
        model.train()
        count, loss_train = 0, 0.0
        for input in train_loader:        
            # Forward Pass
            input = input.to(device=device)
            output = model(input)
            loss    = criterion(output, input)
            loss_train += loss.cpu().detach().numpy()

            # Backward propogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval() 
        count, loss_valid = 0, 0.0
        for input in valid_loader:
            input = input.to(device=device)
            output = model(input)
            loss    = criterion(output, input)  
            loss_valid += loss.cpu().detach().numpy()
            count += 1
        loss_valid /= count
        Loss_valid[epoch] = loss_valid

        trial.report(loss_valid, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return loss_valid


##################################### INPUT #######################################
# Data Parameters
n_halos      = 3674
n_properties = 11
seed         = 4
mass_per_particle = 6.56561e+11

# Training Parameters
num_epochs    = 300
batch_size    = 64

# Architecture Parameters
input_size = 11         # Number of input features 
bottleneck_neurons = 6  # Number of neurons in bottleneck
n_min = 6               # Minimum number of neurons in hidden layers
n_max = 11              # Maximum number of neurons in hidden layers

# Optuna Parameters
n_trials   = 1000 

############################## Start OPTUNA Study ###############################
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
