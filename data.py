import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

################ DATA PARAMETERS #################
n_halos      = 3674
n_properties = 11
seed         = 4
mass_per_particle = 6.56561e+11
filename = "./Halo_Data/Rockstar_z=0.0.txt"

#################################### Select Halos from Data ###################################

# Halo Mask Array
PIDs = np.loadtxt(filename, usecols=41)      # Array of IDs (Halos have ID = -1)
is_halo  = np.array([x == -1 for x in PIDs]) # Boolean array to identify halos from subhalos

# Number of Particles Per Halo >500 
m_vir    = np.loadtxt(filename, skiprows = 16, usecols = 2)[is_halo]
n_particles = m_vir / mass_per_particle
np_mask     = np.array([x>500 for x in n_particles])


#################################  Read halo properties ###############################
m_vir    = np.loadtxt(filename, skiprows = 16, usecols = 2)[is_halo][np_mask]
v_max    = np.loadtxt(filename, skiprows = 16, usecols = 3)[is_halo][np_mask]
v_rms    = np.loadtxt(filename, skiprows = 16, usecols = 4)[is_halo][np_mask]
r_vir    = np.loadtxt(filename, skiprows = 16, usecols = 5)[is_halo][np_mask]
r_s      = np.loadtxt(filename, skiprows = 16, usecols = 6)[is_halo][np_mask]

# Velocities 
v_x      = np.loadtxt(filename, skiprows = 16, usecols = 11)[is_halo][np_mask]
v_y      = np.loadtxt(filename, skiprows = 16, usecols = 12)[is_halo][np_mask]
v_z      = np.loadtxt(filename, skiprows = 16, usecols = 13)[is_halo][np_mask]
v_mag    = np.sqrt((v_x**2) + (v_y**2) + (v_z**2))

# Angular momenta 
J_x      = np.loadtxt(filename, skiprows = 16, usecols = 14)[is_halo][np_mask]
J_y      = np.loadtxt(filename, skiprows = 16, usecols = 15)[is_halo][np_mask]
J_z      = np.loadtxt(filename, skiprows = 16, usecols = 16)[is_halo][np_mask]
J_mag    = np.sqrt((J_x**2) + (J_y**2) + (J_z**2))

# Spin
spin     = np.loadtxt(filename, skiprows = 16, usecols = 17)[is_halo][np_mask]

# b_to_a
b_to_a   = np.loadtxt(filename, skiprows = 16, usecols = 27)[is_halo][np_mask]

# c_to_a
c_to_a   = np.loadtxt(filename, skiprows = 16, usecols = 28)[is_halo][np_mask]

# Ratio of kinetic to potential energies T/|U|
T_U      = np.loadtxt(filename, skiprows = 16, usecols = 37)[is_halo][np_mask]

# Create list for all properties
properties = [m_vir, v_max, v_rms, r_vir, r_s, v_mag, J_mag, spin, b_to_a, c_to_a, T_U]


###################################### NORMALIZE DATA ###################################
def normalize_data(property):
    "This function normalizes the input data"
    mean = np.mean(property)
    std  = np.std(property)
    normalized = (property - mean)/std
    
    return normalized

# Take log10 of m_vir and J_mag
m_vir_log  = np.log10(m_vir+1)
J_mag_log  = np.log10(J_mag+1)

# New properties array
properties = np.array([m_vir_log, v_max, v_rms, r_vir, r_s, v_mag, J_mag_log, spin, b_to_a, c_to_a, T_U])

# Normalize the properties
norm_properties = np.zeros((len(properties), len(properties[0])), dtype = np.float32)
for i in range(len(properties)):
    norm_properties[i]  = normalize_data(properties[i])
    
# Reshape data
halo_data = norm_properties.reshape(n_halos, n_properties)

# Convert to torch tensor
halo_data = torch.tensor(halo_data, dtype=torch.float)


# Make custom torch dataset
class make_Dataset(Dataset):
    
    def __init__(self, name, seed, n_halos, halo_data):
         
        # shuffle the halo number (instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342)
        # n_halos = 495293
        # halo_data shape = (n_halo, number of properties) = (495293, 11)
        
        np.random.seed(seed)
        indexes = np.arange(n_halos)
        np.random.shuffle(indexes)
        
        # Divide the dataset into train, valid, and test sets
        if   name=='train':  size, offset = int(n_halos*0.8), int(n_halos*0.0)
        elif name=='valid':  size, offset = int(n_halos*0.1), int(n_halos*0.8)
        elif name=='test' :  size, offset = int(n_halos*0.1), int(n_halos*0.9)
        else:                raise Exception('Wrong name!')
        
        self.size   = size
        self.input  = torch.zeros((size, 11), dtype=torch.float) # Each input has a shape of (11, 1) (flattened)
        
        # do a loop over all elements in the dataset
        for i in range(size):
            
            # find the halo index (shuffled)
            j = indexes[i+offset]

            # load data
            self.input [i] = halo_data[j]
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx]
    
########################### CREATE DATASETS ############################

#This function creates datasets for train, valid, test
def create_datasets(seed, n_halos, halo_data, batch_size):
    
    train_Dataset = make_Dataset('train', seed, n_halos, halo_data)
    valid_Dataset = make_Dataset('valid', seed, n_halos, halo_data)
    test_Dataset  = make_Dataset('test',  seed, n_halos, halo_data)
    
    return train_Dataset, valid_Dataset, test_Dataset
