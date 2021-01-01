import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time


# This function reads the rockstar file
def read_data(f_rockstar):
    # Halo Mask Array
    PIDs = np.loadtxt(filename, usecols=41)       # Array of IDs (Halos have ID = -1)
    is_halo  = np.array([x == -1 for x in PIDs])  # Conditional array to identify halos from subhalos

    # Number of Particles Per Halo >500 
    mass_per_particle = 6.56561e+11
    m_vir    = np.loadtxt(filename, skiprows = 16, usecols = 2)[is_halo]
    n_particles = m_vir / mass_per_particle
    np_mask     = np.array([x>500 for x in n_particles])

    # Get the number of halos and properties
    n_halos = np.size(m_vir[np_mask])
    n_properties = 11

    #################################### LOAD DATA ###################################
    # Define container for data 
    data = np.zeros((n_halos, n_properties), dtype=np.float32)

    #m_vir
    data[:,0] = np.loadtxt(filename, skiprows = 16, usecols = 2)[is_halo][np_mask]

    #v_max
    data[:,1] = np.loadtxt(filename, skiprows = 16, usecols = 3)[is_halo][np_mask]

    # v_rms
    data[:,2] = np.loadtxt(filename, skiprows = 16, usecols = 4)[is_halo][np_mask]

    # r_vir
    data[:,3] = np.loadtxt(filename, skiprows = 16, usecols = 5)[is_halo][np_mask]

    # r_s
    data[:,4] = np.loadtxt(filename, skiprows = 16, usecols = 6)[is_halo][np_mask]

    # Velocities 
    v_x      = np.loadtxt(filename, skiprows = 16, usecols = 11)[is_halo][np_mask]
    v_y      = np.loadtxt(filename, skiprows = 16, usecols = 12)[is_halo][np_mask]
    v_z      = np.loadtxt(filename, skiprows = 16, usecols = 13)[is_halo][np_mask]
    v_mag    = np.sqrt((v_x**2) + (v_y**2) + (v_z**2))
    data[:,5] = v_mag

    # Angular momenta 
    J_x      = np.loadtxt(filename, skiprows = 16, usecols = 14)[is_halo][np_mask]
    J_y      = np.loadtxt(filename, skiprows = 16, usecols = 15)[is_halo][np_mask]
    J_z      = np.loadtxt(filename, skiprows = 16, usecols = 16)[is_halo][np_mask]
    J_mag    = np.sqrt((J_x**2) + (J_y**2) + (J_z**2))
    data[:,6] = J_mag

    # Spin
    data[:,7] = np.loadtxt(filename, skiprows = 16, usecols = 17)[is_halo][np_mask]

    # b_to_a
    data[:,8] = np.loadtxt(filename, skiprows = 16, usecols = 27)[is_halo][np_mask]

    # c_to_a
    data[:,9] = np.loadtxt(filename, skiprows = 16, usecols = 28)[is_halo][np_mask]

    # Ratio of kinetic to potential energies T/|U|
    data[:,10] = np.loadtxt(filename, skiprows = 16, usecols = 37)[is_halo][np_mask]

    ############################# NORMALIZE DATA ##############################
    # This function normalizes the input data
    def normalize_data(data):
        
        n_halos = data.shape[0]
        n_properties = data.shape[1]
        data_norm = np.zeros((n_halos, n_properties), dtype=np.float32)
        
        for i in range(n_properties):
            mean = np.mean(data[:,i])
            std  = np.std(data[:,i])
            normalized = (data[:,i] - mean)/std
            data_norm[:,i] = normalized
        return(data_norm)

    # Take log10 of m_vir and J_mag
    data[:,0]  = np.log10(data[:,0]+1)
    data[:,6]  = np.log10(data[:,6]+1)

    # Normalize each property
    halo_data = normalize_data(data)

    # Convert to torch tensor
    halo_data = torch.tensor(halo_data, dtype=torch.float)
    
    return halo_data


###################################### Create Datasets ###################################
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
        
        # Get the data
        halo_data = read_data(f_rockstar)
        
        # do a loop over all elements in the dataset
        for i in range(size):
            j = indexes[i+offset]          # find the halo index (shuffled)
            self.input [i] = halo_data[j]  # load data
            
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx]

    
#This function creates datasets for train, valid, test
def create_datasets(seed, n_halos, halo_data, batch_size):
    
    train_Dataset = make_Dataset('train', seed, n_halos, halo_data)
    valid_Dataset = make_Dataset('valid', seed, n_halos, halo_data)
    test_Dataset  = make_Dataset('test',  seed, n_halos, halo_data)
    
    return train_Dataset, valid_Dataset, test_Dataset
