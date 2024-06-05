# Imports all the necessary packages
import pandas as pd
import rdkit
import rdkit.Chem
import numpy as np
import random
import os
import ast
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import global_add_pool, GCNConv
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ExponentialLR

from sklearn.model_selection import KFold

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray import train

# only want to use 5 cpu cores
ray.init(num_cpus=5)

from collections import deque

# For creating the parity plot
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# My data only consists of 9 different elements. These are given in the dictionary below, along with their atom number.
elements_allowed = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def extract_mean_std(path_to_code, number_of_temp, min_interval):
    """
    Extracts the mean and standard deviation for the training data. 
    """
    with open(f"{path_to_code}/Data/Min interval of {min_interval} ºC/{number_of_temp} temperatures/Pressure Overview (log).txt", 'r') as file:
        data = file.readlines()
        for line in data: 
            if line.startswith("Mean pressure in training data"):
                mean = float(line.split(":")[1].strip())
            if line.startswith("Standard deviation of pressure in training data"):
                std = float(line.split(":")[1].strip())

    return mean, std


def extract_VLE_data(excel_filename, path_to_code):
    """
    Extracts the data that will be used for training and validating the model.
    """

    # Locates the file for the training data.
    excel_file_VLE = pd.ExcelFile(f"{path_to_code}/Data/Combined model/{excel_filename}")

    # Locates the file for the SMILES notation.
    smiles_converter = pd.read_excel(f"{path_to_code}/Data/Amines_looked_up.xlsx")

    # Only want the columns "CAS No" and "SMILES"
    smiles_converter = smiles_converter[["CAS No", "SMILES"]]

    # Dictionary to store the data. The key is the SMILES notation and the value is the VLE data.
    dev_dict = {}

    for sheet in excel_file_VLE.sheet_names:
        # Get the correct SMILES notation for the compound.
        smiles_value = smiles_converter.loc[smiles_converter["CAS No"] == sheet, "SMILES"].values[0]
        dev_dict[smiles_value] = pd.read_excel(excel_file_VLE, sheet_name=sheet)

    return dev_dict


def create_parity_plot(y_hat_values, target_values, train_test_val, epoch=None, seed=None, split_ratio = None, cross_validation=False, fold_number=None, final_model=False):
    # Extract the values for the amines and water fractions.
    y_hat_amines = y_hat_values[:, 0]
    y_hat_water = y_hat_values[:, 1]
    target_amines = target_values[:, 0]
    target_water = target_values[:, 1]

    # For plotting the parity line. To do this, we need to find the minimum and maximum values 
    # (can either be from the actual y value or the predicted value).
    min_val_amines = min(min(y_hat_amines), min(target_amines))
    max_val_amines = max(max(y_hat_amines), max(target_amines))

    min_val_water = min(min(y_hat_water), min(target_water))    
    max_val_water = max(max(y_hat_water), max(target_water))

    X_val_list_amines = [min_val_amines, max_val_amines]
    X_val_list_water = [min_val_water, max_val_water]
    y_val_list_amines = [min_val_amines, max_val_amines]
    y_val_list_water = [min_val_water, max_val_water]

    ### Plotting the parity plot for the amines ###

    # Create a line between the minimum value and the maximum value.
    plt.figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45)
    # Plots the actual values against the predicted values of the model.
    plt.scatter(target_amines, y_hat_amines, c=CB_color_cycle[0], label="Amines", alpha=0.7, edgecolors='w', s=40)
    # Plots the parity line.
    plt.plot(X_val_list_amines, y_val_list_amines, 'k--')
    plt.legend()
    plt.xlabel('True', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.title(f"Parity Plot - {train_test_val}", fontsize=18)

    # Saves the figure in a folder
    plt.tight_layout()

    if final_model:
        plt.savefig(f"Data/Best combined model/Final evaluation/Amine/Parity Plot - {train_test_val}", dpi=300)

    elif not cross_validation:
        if not os.path.exists(f"Data/Combined model/GCN models"):
            os.mkdir(f"Data/Combined model/GCN models")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}/Amine"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}/Amine")

        plt.savefig(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}/Amine/Parity Plot - {train_test_val}", dpi=300)

    else:
        if not os.path.exists(f"Data/Combined model/GCN Cross Validation"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}/Amine"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}/Amine")

        plt.savefig(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}/Amine/Parity Plot - {train_test_val}", dpi=300)
    plt.close()

    ### Plotting the parity plot for water ###

    # Create a line between the minimum value and the maximum value.
    plt.figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45)
    # Plots the actual values against the predicted values of the model.
    # Plots the actual values against the predicted values of the model.
    plt.scatter(target_water, y_hat_water, c=CB_color_cycle[0], label="Water", alpha=0.7, edgecolors='w', s=40)
    # Plots the parity line.
    plt.plot(X_val_list_water, y_val_list_water, 'k--', label='Parity Line')
    plt.legend()
    plt.xlabel('True', fontsize=18)
    plt.ylabel('Predicted', fontsize=18)
    plt.title(f"Parity Plot - {train_test_val}", fontsize=18)

    # Saves the figure in a folder
    plt.tight_layout()

    if final_model:
        plt.savefig(f"Data/Best combined model/Final evaluation/Water/Parity Plot - {train_test_val}", dpi=300)

    elif not cross_validation:
        if not os.path.exists(f"Data/Combined model/GCN models"):
            os.mkdir(f"Data/Combined model/GCN models")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}")

        if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}/Water"):
            os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}/Water")

        plt.savefig(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Parity plots/Epoch {epoch}/Water/Parity Plot - {train_test_val}", dpi=300)

    else:
        if not os.path.exists(f"Data/GCN Cross Validation"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}")

        if not os.path.exists(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}/Water"):
            os.mkdir(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}/Water")

        plt.savefig(f"Data/Combined model/GCN Cross Validation/Parity plots/Fold {fold_number}/Epoch {epoch}/Water/Parity Plot - {train_test_val}", dpi=300)
    plt.close()


def shuffle_data(dictionary_data):
    """
    This function shuffles the whole dataset and convert the temperature to Celsius.
    """

    SMILES = []
    x_amine = []
    x_water = []
    y_water = []
    y_amine = []
    T = []
    P_tot = []

    for key, data in dictionary_data.items():
        for index, row in data.iterrows():
            SMILES.append(key)
            x_amine.append(row["x_amine"])
            x_water.append(row["x_water"])
            y_water.append(row["y_water"])
            y_amine.append(row["y_amine"])
            T.append(row["T"] - 273.15)
            P_tot.append(row["P"])

    # Shuffle the indices
    indices = list(range(len(SMILES)))
    random.shuffle(indices)

    # Shuffle the data
    SMILES = [SMILES[i] for i in indices]
    x_amine = [x_amine[i] for i in indices]
    x_water = [x_water[i] for i in indices]
    y_water = [y_water[i] for i in indices]
    y_amine = [y_amine[i] for i in indices]
    T = [T[i] for i in indices]
    P_tot = [P_tot[i] for i in indices]

    return SMILES, x_water, x_amine, y_water, y_amine, T, P_tot


# Function to set seed for reproducibility
def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)  # Set numpy seed
    torch.manual_seed(seed_value)  # Set torch seed
    random.seed(seed_value)  # Set python random seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Set python environment seed


# Converts the SMILES notation to a graph
def smiles2graph(sml):
    """
    This code is based on the code from the book "Deep Learning for Molecules and Materials" byAndrew D White.
    This function will return the graph of a molecule based on the SMILES string.
    """
    m = rdkit.Chem.MolFromSmiles(sml)
    m = rdkit.Chem.AddHs(m)
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }

    # The length of the adjacency matrix should be NxN, where N is the number of atoms in the molecule.
    N = len(list(m.GetAtoms()))
    nodes = np.zeros((N, len(elements_allowed)))
    lookup = list(elements_allowed.keys())
    for i in m.GetAtoms():
        # If an atom is present in our molecule,
        nodes[i.GetIdx(), lookup.index(i.GetAtomicNum())] = 1

    adj = np.zeros((N, N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning("Ignoring bond order" + order)
        adj[u, v] = 1
        adj[v, u] = 1
    # We want the diagonal in the matrix to be 1.
    adj += np.eye(N)
    return nodes, adj


def merge_batch(batch):
    """
    This function merge the graphs in the batch into a larger batch by concatenating node features and computing new
    edge indices. Batch is on the format [(graph1, target1), (graph2, target2), ...], and the output is on the format
    (merged_nodes, merged_edge_indices, merged_batch_mapping), merged_targets, merged_temperatures.

    Further, merged_nodes is on the format [node1, node2, ...], merged_edge_indices is on the format [edge1, edge2, ...]
    and merged_batch_mapping is on the format [0, 0, ..., 1, 1, ...], where the length of the lists are the number of
    nodes and edges in the merged graph, respectively. The merged_targets is on the format [target1, target2, ...].
    The merged_temperatures is on the format [temperature1, temperature2, ...].
    """

    # Separate out nodes, adjacency matrices, scalar tensors, and temperatures
    nodes_list = [item[0][0] for item in batch]
    adj_list = [item[0][1] for item in batch]
    x_water_tensor = [item[1] for item in batch]
    x_amine_tensor = [item[2] for item in batch]
    y_water_tensor = [item[3] for item in batch]
    y_amine_tensor = [item[4] for item in batch]
    temperatures = [item[5] for item in batch]
    P_tot = [item[6] for item in batch]

    # Placeholder for the combined nodes
    merged_nodes = []
    # Placeholder for the combined edge indices (plural for index) -> says something about connections
    edge_indices = []
    # Placeholder for the batch mapping -> says something about which node that belongs to which graph! Important for
    # tying things together.
    batch_mapping = []

    # This will keep track of the current node index offset in the combined graph -> how much we must "shift" the
    # edge matrix
    current_node_index = 0

    # Iterates over all the graphs
    for idx, (nodes, adj) in enumerate(zip(nodes_list, adj_list)):
        # Extracts the number of nodes in the current graph.
        num_nodes = nodes.shape[0]

        # Add the current graph's nodes to the merged node list. Can append as usual, since we want to add rows.
        merged_nodes.append(nodes)

        # Converts the adjacency matrix to the correct edge index format required for the GCN layer. In other words, we
        # find the edges in the adjacency matrix, and offset them by the current node index.
        edges = adj.nonzero().t()
        edges = edges + current_node_index

        # Add the current graph's edges to the combined list
        edge_indices.append(edges)

        # Create the batch mapping for the current graph
        batch_mapping.extend([idx] * num_nodes)

        # Update the node index offset
        current_node_index += num_nodes

    # Convert the merged node list to a tensor
    merged_nodes_tensor = torch.cat(merged_nodes, dim=0)

    # Convert the edge index lists to a single tensor
    merged_edge_indices_tensor = torch.cat(edge_indices, dim=1)

    # Convert the batch mapping to a tensor
    batch_mapping_tensor = torch.tensor(batch_mapping, dtype=torch.long)

    # Convert x_water_tensor, x_amine_tensor, y_water_tensor, y_amine_tensor, temperature_tensor and P_tot_tensor to single tensors.
    x_water_tensor = torch.stack(x_water_tensor)
    x_amine_tensor = torch.stack(x_amine_tensor)
    y_water_tensor = torch.stack(y_water_tensor)
    y_amine_tensor = torch.stack(y_amine_tensor)
    temperature_tensor = torch.stack(temperatures)
    P_tot_tensor = torch.stack(P_tot)

    return (merged_nodes_tensor, merged_edge_indices_tensor, batch_mapping_tensor), x_water_tensor, x_amine_tensor, y_water_tensor, y_amine_tensor, temperature_tensor, P_tot_tensor


class MolecularDataset(Dataset):
    """
    This class is needed to create our dataset (on the Dataset format).
    The class inherits from the Dataset class. Input is on format X and T, where X is the SMILES notation and T is the
    temperature. 
    """

    def __init__(self, data):
        # Initializes the features and targets. Is our constructor.
        self.SMILES = data[0]
        self.x_water = data[1]
        self.x_amine = data[2]
        self.y_water = data[3]
        self.y_amine = data[4]
        self.temperature = data[5]
        self.P_tot = data[6]

    def __len__(self):
        # Returns the length of the dataset
        return len(self.SMILES)

    def __getitem__(self, idx):
        # Extract the SMILES value of the molecule, calculate the graphs based on that SMILE, and then return this
        # value along with the corresponding target value.
        SMILES_one_molecule = self.SMILES[idx]
        nodes, adj = smiles2graph(SMILES_one_molecule)

        # Convert nodes and adj to tensors, assuming they are NumPy arrays returned from smiles2graph
        nodes_tensor = torch.tensor(nodes, dtype=torch.float32)
        adj_tensor = torch.tensor(adj, dtype=torch.float32)

        # Convert x_water, x_amine, y_water, y_amine, temperature and pressure to tensors.
        x_water_tensor = torch.tensor(self.x_water[idx], dtype=torch.float32)
        x_amine_tensor = torch.tensor(self.x_amine[idx], dtype=torch.float32)
        y_water_tensor = torch.tensor(self.y_water[idx], dtype=torch.float32)
        y_amine_tensor = torch.tensor(self.y_amine[idx], dtype=torch.float32)
        temperature_tensor = torch.tensor(self.temperature[idx], dtype=torch.float32)
        P_tot_tensor = torch.tensor(self.P_tot[idx], dtype=torch.float32)

        return (nodes_tensor, adj_tensor), x_water_tensor, x_amine_tensor, y_water_tensor, y_amine_tensor, temperature_tensor, P_tot_tensor


class AntoineEquationLayer(nn.Module):
    def __init__(self):
        """
        Initialize the custom layer. 
        """
        super(AntoineEquationLayer, self).__init__()

    def forward(self, x_in, temperature):
        """
        Forward pass of the layer using the Antoine equation.
        x_in is expected to have three elements representing A, B, and C coefficients. Note that x_in will have the
        shape (batch_size, 3). This layer also expect a temperature as input.
        """
    
        A, B, C = x_in[:, 0], x_in[:, 1], x_in[:, 2]

        # Here, P will be the pressure in mmHg! 
        log_P = A - B / (temperature + C)

        return log_P


class NRTLEquationLayer(nn.Module):
    def __init__(self):
        """
        Initialize the custom layer. 
        """
        super(NRTLEquationLayer, self).__init__()

    def forward(self, x_in, temperature, mole_frac_amine, mole_frac_water):
        """
        x_in have four elements representing alpha, b_12, and b_21 coefficients.
        """

        # Extract the interaction parameters. 
        alpha, b_12, b_21 = x_in[:, 0], x_in[:, 1], x_in[:, 2]

        # Sets alpha to be constant at 0.3 -> Better to let it vary. 
        #alpha = 0.3
        alpha_min = 0.2
        alpha_max = 0.47

        # Scales alpha from 0 to 1
        #alpha = torch.sigmoid(alpha)
        # Scales alpha from [alpha_min, alpha_max]
        #alpha = alpha_min + (alpha_max - alpha_min) * alpha

        # Calculate alpha to get it in that range (from spt-nrtl paper)
        alpha = alpha_min * (1 + torch.sigmoid(alpha)/10 * ((alpha_max/alpha_min)))

        # Gas constant in mmHg
        R = 62.36367 # mmHg L / mol K

        # The temperature in Kelvin
        temperature = temperature + 273.15

        # Calculate tau 
        tau_12 = b_12/(R*temperature)
        tau_21 = b_21/(R*temperature)

        # Calculate G
        G_12 = torch.exp(-tau_12 * alpha)
        G_21 = torch.exp(-tau_21 * alpha)

        # Calculate the activity coefficients
        ln_gamma_amine = mole_frac_water**2 * (tau_21 * (G_21 / (mole_frac_amine + mole_frac_water * G_21) )**2 + (tau_12 * G_12)/( mole_frac_water + mole_frac_amine*G_12 )**2)
        ln_gamma_water = mole_frac_amine**2 * (tau_12 * (G_12 / (mole_frac_water + mole_frac_amine * G_12) )**2 + (tau_21 * G_21)/( mole_frac_amine + mole_frac_water*G_21 )**2)

        return ln_gamma_amine, ln_gamma_water
    

class GNN_Saturation_Pressure(nn.Module):
    """
    This class defines the structure of the model. The model will be a graph neural network (GNN) model. The model
    inherits from the nn.Module class, which is the base class for all neural network modules in PyTorch. 
    """

    def __init__(self, config, num_of_features):
        # Defines the structure of the model. 
        super().__init__()

        # Set the activation function.
        if config["activation_function"] == "relu":
            self.activation = nn.ReLU()

        elif config["activation_function"] == "sigmoid":
            self.activation = nn.Sigmoid()

        elif config["activation_function"] == "tanh":
            self.activation = nn.Tanh()

        # Initialize GCN layers and activations as ModuleList
        self.GCN_layers = nn.ModuleList()
        self.GCN_activations = nn.ModuleList()

        # Adds the GCN layers and the activation functions to the model.
        for i in range(best_config["num_gcn_layers"]):
            self.GCN_layers.append(GCNConv(num_of_features, best_config["GCN_output_per_layer"][i]))
            self.GCN_activations.append(self.activation)
            num_of_features = best_config["GCN_output_per_layer"][i]

        # Adds the global pooling layer.
        self.global_pooling = global_add_pool
        self.global_pooling_activation = self.activation

        # Initialize dense layers and activations as ModuleList
        self.dense_layers = nn.ModuleList()
        self.dense_activations = nn.ModuleList()
        self.dropout = nn.ModuleList()

        # Adds the dense layers and the activation functions to the model.
        for i in range(best_config["num_hidden_layers"]):
            self.dense_layers.append(nn.Linear(num_of_features, best_config["hidden_neurons_per_layer"][i]))
            self.dense_activations.append(self.activation)
            self.dropout.append(nn.Dropout(p=best_config["dropout_rate"]))
            num_of_features = best_config["hidden_neurons_per_layer"][i]

        # Adds the Antoine coefficients layer.
        self.antonine_coeff = nn.Linear(num_of_features, 3)
        
        # Create the Antinone equation layer separatly as it is not possible to pass additional arguments to the
        # Sequential class. However, we need to pass the temperature to the Antoine equation layer.
        self.antonine_layer = AntoineEquationLayer()

    def forward(self, x, edge_indices, batch_mapping, temperature, mean, std):
        # Defines the forward pass of the model. This is where the data is input to the model.

        #print(f"Input shape: {x.shape}")
        # Iterates over all the GCN layers and the activation functions.
        for layer, act in zip(self.GCN_layers, self.GCN_activations):
            # Performs the message passing and then applies the activation function. Edge index says which atoms 
            # that are connected. 
            x = act(layer(x, edge_indices))
            #print(f"After GCN Layer {layer}: {x.shape}")

        # Apply global pooling. Here we need batch_mapping to map which atoms that belongs to which molecule.
        x = self.global_pooling(x, batch_mapping)
        x = self.global_pooling_activation(x)
        #print(f"After Global Pooling: {x.shape}")

        # Iterates over all the dense layers and the activation functions.
        for layer, act, drop in zip(self.dense_layers, self.dense_activations, self.dropout):
            # Applies the dense layer and then the activation function.
            x = act(layer(x))
            # Apply dropout
            x = drop(x)
            #print(f"After Dense Layer {layer}: {x.shape}")

        # Apply the Antoine coefficients layer
        x = self.antonine_coeff(x)
        #print(f"After Antoine Coefficients Layer: {x.shape}")

        log_P = self.antonine_layer(x, temperature)
        #print(f"Final Output Shape: {log_P.shape}")

        # Scale the output
        log_P = (log_P - mean) / std

        return log_P


class GNN_Activity_coeff(nn.Module):
    """
    This class defines the structure of the model. The model will be a graph neural network (GNN) model. The model
    inherits from the nn.Module class, which is the base class for all neural network modules in PyTorch. 
    """

    def __init__(self, config, num_of_features):
        # Defines the structure of the model. 
        super().__init__()

        # Set the activation function.
        if config["activation_function"] == "relu":
            self.activation = nn.ReLU()

        elif config["activation_function"] == "sigmoid":
            self.activation = nn.Sigmoid()

        elif config["activation_function"] == "tanh":
            self.activation = nn.Tanh()

        # Initialize GCN layers and activations as ModuleList
        self.GCN_layers = nn.ModuleList()
        self.GCN_activations = nn.ModuleList()

        # Adds the GCN layers and the activation functions to the model.
        for i in range(best_config["num_gcn_layers"]):
            self.GCN_layers.append(GCNConv(num_of_features, best_config["GCN_output_per_layer"][i]))
            self.GCN_activations.append(self.activation)
            num_of_features = best_config["GCN_output_per_layer"][i]

        # Adds the global pooling layer.
        self.global_pooling = global_add_pool
        self.global_pooling_activation = self.activation

        # Initialize dense layers and activations as ModuleList
        self.dense_layers = nn.ModuleList()
        self.dense_activations = nn.ModuleList()
        self.dropout = nn.ModuleList()

        # Adds the dense layers and the activation functions to the model.
        for i in range(best_config["num_hidden_layers"]):
            self.dense_layers.append(nn.Linear(num_of_features, best_config["hidden_neurons_per_layer"][i]))
            self.dense_activations.append(self.activation)
            self.dropout.append(nn.Dropout(p=best_config["dropout_rate"]))
            num_of_features = best_config["hidden_neurons_per_layer"][i]

        # Adds the NRTL coefficients layer.
        self.NRTL_coeff = nn.Linear(num_of_features, 3)
        
        # Initilize the NRTRL layer
        self.nrtl_layer = NRTLEquationLayer()

    def forward(self, x, edge_indices, batch_mapping, temperature, mean, std, mole_frac_amine, mole_frac_water):
        # Defines the forward pass of the model. This is where the data is input to the model.

        #print(f"Input shape: {x.shape}")
        # Iterates over all the GCN layers and the activation functions.
        for layer, act in zip(self.GCN_layers, self.GCN_activations):
            # Performs the message passing and then applies the activation function. Edge index says which atoms 
            # that are connected. 
            x = act(layer(x, edge_indices))
            #print(f"After GCN Layer {layer}: {x.shape}")

        # Apply global pooling. Here we need batch_mapping to map which atoms that belongs to which molecule.
        x = self.global_pooling(x, batch_mapping)
        x = self.global_pooling_activation(x)
        #print(f"After Global Pooling: {x.shape}")

        # Iterates over all the dense layers and the activation functions.
        for layer, act, drop in zip(self.dense_layers, self.dense_activations, self.dropout):
            # Applies the dense layer and then the activation function.
            x = act(layer(x))
            # Apply dropout
            x = drop(x)
            #print(f"After Dense Layer {layer}: {x.shape}")

        # Apply the Antoine coefficients layer
        x = self.NRTL_coeff(x)
        #print(f"After NRTL Coefficients Layer: {x.shape}")

        ln_gamma_amine, ln_gamma_water = self.nrtl_layer(x, temperature, mole_frac_amine, mole_frac_water)
        #print(f"Final Output Shape: {log_P.shape}")

        return ln_gamma_amine, ln_gamma_water
    

def saturation_pressure_water(T):
    """
    This function calculates the saturation pressure of water at a given temperature. These formulas are based on the
    Antoine equation recived from the Dortmund Data Bank.
    """

    # Initialize a tensor for P with the same shape as T and fill it with NaNs.
    P = torch.full_like(T, float('nan'))

    # Condition for temperature range 1 to 100 degrees Celsius
    condition1 = (T > 1) & (T < 100)
    A1 = 8.07131
    B1 = 1730.63
    C1 = 233.426
    P[condition1] = 10**(A1 - B1 / (T[condition1] + C1))

    # Condition for temperature range 100 to 374 degrees Celsius
    condition2 = (T >= 100) & (T < 374)
    A2 = 8.14019
    B2 = 1810.94
    C2 = 244.485
    P[condition2] = 10**(A2 - B2 / (T[condition2] + C2))

    return P
    

class VLE_model(nn.Module):
    """
    This class defines the structure of the model. The model will be a graph neural network (GNN) model. The model
    inherits from the nn.Module class, which is the base class for all neural network modules in PyTorch. 
    """

    def __init__(self, GCN_saturation, mean_gcn, std_gcn, best_config, num_of_features):
        # Defines the structure of the model. 
        super().__init__()

        # Initialize the GCN saturation model with the pre-trained model. Also initilize the mean and standard deviation for scaling.
        self.GCN_saturation = GCN_saturation
        self.mean_gcn = mean_gcn
        self.std_gcn = std_gcn

        # Initilize the NRTL model
        self.NRTL_model = GNN_Activity_coeff(best_config, num_of_features)


    def forward(self, batch):
        """ Defines the forward pass of the model. This is where the data is input to the model. """

        # Unpack the batch
        (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

        # Calculate the predictions of the GCN model
        log_P_normalized = self.GCN_saturation(x, edge_indices, batch_mapping, temperature, self.mean_gcn, self.std_gcn)
        log_P = log_P_normalized * self.std_gcn + self.mean_gcn
        P_amine = 10**log_P
        ln_P_amine = torch.log(P_amine)

        # Convert total pressure from Pa to mmHg
        P_tot = P_tot / 133.322

        # Take the natural logarithm of the total pressure, water fraction and amine fraction
        ln_P_tot = torch.log(P_tot)
        ln_x_water = torch.log(x_water)
        ln_x_amine = torch.log(x_amine)

        # Calculate the predictions of the NRTL model (activity coefficients)
        ln_gamma_amine, ln_gamma_water = self.NRTL_model(x, edge_indices, batch_mapping, temperature, self.mean_gcn, self.std_gcn, x_amine, x_water)

        # Calculate the saturation pressure of water
        P_water = saturation_pressure_water(temperature)
        ln_P_water = torch.log(P_water)

        # Calculate ln_y_amine and ln_y_water
        ln_y_amine_pred = ln_x_amine + ln_gamma_amine + ln_P_amine - ln_P_tot
        ln_y_water_pred = ln_x_water + ln_gamma_water + ln_P_water - ln_P_tot

        # Want to return the predictions of the amine and water fractions. 
        sum_ln_y = ln_y_amine_pred + ln_y_water_pred

        return torch.stack((ln_y_amine_pred, ln_y_water_pred, sum_ln_y), dim=1)


def plot_learning_curve(cross_validation, seed=None, split_ratio=None):
    folder = None
    if cross_validation:
        folder = "GCN Cross Validation"

    else: 
        folder = "GCN models"

    # Want to extract the average data created by "train_model_with_CV" from the Excel file. Was used to find the 
    # lowest validation error. 
    if seed is None:
        df_train = pd.read_excel(f"Data/Combined model/{folder}/Train_error.xlsx")
        df_validation = pd.read_excel(f"Data/Combined model/{folder}/Validation_error.xlsx")
    else:
        df_train = pd.read_excel(f"Data/Combined model/{folder}/Split ratio = {split_ratio}/{seed}/Train_error.xlsx")
        df_validation = pd.read_excel(f"Data/Combined model/{folder}/Split ratio = {split_ratio}/{seed}/Validation_error.xlsx")
    average_train_error = df_train["Average"].tolist()
    average_validation_error = df_validation["Average"].tolist()

    # Print out the lowest validation error
    lowest_error = min(average_validation_error)
    print(lowest_error)

    # Save the lowest error in a text file.
    if seed is None:
        with open(f"Data/Combined model/{folder}/Lowest_error.txt", "w") as file:
            file.write(str(lowest_error))
    else:
        with open(f"Data/Combined model/{folder}/Split ratio = {split_ratio}/{seed}/Lowest_error.txt", "w") as file:
            file.write(str(lowest_error))

    # Want to plot this to visualize the optimal number of epochs. x-axis is the epoch number and y-axis the validation error.
    plt.plot(average_validation_error, label="Validation Error")
    plt.plot(average_train_error, label="Training Error")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Cross-validation error [MSE]", fontsize=18)
    plt.title("Error for each epoch", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Want to rotate the x-axis labels by 90 degrees so they don't overlap
    plt.xticks(rotation=45)
    # include grids for every 50 epochs
    plt.xticks(np.arange(50, len(average_validation_error) + 50, 50))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if seed is None:
        plt.savefig(f"Data/Combined model/{folder}/Validation_error_plot.png", dpi=300)
    else:
        plt.savefig(f"Data/Combined model/{folder}/Split ratio = {split_ratio}/{seed}/Validation_error_plot.png", dpi=300)

    # Note that if the figure is not closed, one will get a plot of the learning curves for all seeds. 
    plt.close()

    # Also want a plot that don't include the first 100 epochs.
    # Do not want to plot the first 100 elements
    plt.figure()
    average_validation_error = average_validation_error[100:]
    average_train_error = average_train_error[100:]
    # I want the x-axis to start at 100
    x_axis = list(range(100, len(average_validation_error) + 100))
    plt.plot(x_axis, average_validation_error, label="Validation Error")
    plt.plot(x_axis, average_train_error, label="Training Error")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Error [MSE]", fontsize=18)
    plt.title("Learning curve", fontsize=18)
    # Want the x-axis and y-axis numbers to be larger so it becomes easier to read
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Want to rotate the x-axis labels by 90 degrees so they don't overlap
    plt.xticks(rotation=45)
    # include grids for every 50 epochs, including the grid at epoch 50
    plt.xticks(np.arange(100, len(average_validation_error) + 100, 50))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if seed is None:
        plt.savefig(f"Data/Combined model/{folder}/Validation_error_plot_excluding_first_100.png", dpi=300)
    else:
        plt.savefig(f"Data/Combined model/{folder}/Split ratio = {split_ratio}/{seed}/Validation_error_plot_excluding_first_100.png", dpi=300)    
    plt.close()


def train_with_CV(best_config, number_of_folds=10, number_of_epochs=300):
    """
    Main execution function.
    """

    # Sets a seed for splitting data
    seed = 42
    set_seed(seed)

    # Path to the folder where the code is located.
    path_to_code = os.path.dirname(os.path.realpath(__file__))

    # Extracts the mean and standard deviation for scaling. We use the same scaling as before. 
    mean_gcn, std_gcn = extract_mean_std(path_to_code, best_config["number_of_temp"], best_config["min_interval"])

    # Define the number of features for the GCN layer
    num_of_features = len(elements_allowed)

    # Extracts the development data as a dictionary. The dictionary contains the SMILES notation for all the compounds as keys,
    # and the corresponding VLE data as values. 
    dev_dict = extract_VLE_data("VLE_experimental_y_train.xlsx", path_to_code)

    # Initialize the index to keep track of the fold.
    index = 1

    # Perform cross validation. Can not treat the data more before loop, as we want to split on amines to avoid leakage between
    # the training and validation data.
    cross_validation = KFold(n_splits=number_of_folds)
    for train_index, test_index in cross_validation.split(dev_dict):

        seed = 42

        # Sets a seed for for reinitializing the parameters in the current split. 
        nan_encountered = True
        while nan_encountered: 
            set_seed(seed)
            nan_encountered = False
            print(seed)
            seed += 1

            train_data = {}
            validation_data = {}

            # Extract the training and validation data based on the indices. 
            for i in train_index:
                train_data[list(dev_dict.keys())[i]] = dev_dict[list(dev_dict.keys())[i]]

            for i in test_index:
                validation_data[list(dev_dict.keys())[i]] = dev_dict[list(dev_dict.keys())[i]]

            # Shuffle the training and validation data. Returns lists where the same index corresponds to the same molecule.
            shuffled_train_data = shuffle_data(train_data)
            shuffled_validation_data = shuffle_data(validation_data)

            # Convert smiles to graphs and prepare the data for the model.
            train_dataset_molecular = MolecularDataset(shuffled_train_data)
            validation_dataset_molecular = MolecularDataset(shuffled_validation_data)
            
            # Creating a DataLoader which handles batching and shuffling. collate_fn treat one batch of train_dataset
            # to get it on the right format for the GCN layer (converts graphs and adjacency matrices and maps them).
            train_loader = DataLoader(train_dataset_molecular, batch_size=best_config["size_of_batch"], collate_fn=merge_batch)
            validation_loader = DataLoader(validation_dataset_molecular, batch_size=best_config["size_of_batch"], collate_fn=merge_batch)

            # Initilize and load the best gcn model for predicting amine pressure from earlier training.
            gcn_saturation = GNN_Saturation_Pressure(best_config, num_of_features)
            gcn_saturation.load_state_dict(torch.load(f"Data/Best amine model/Best_model.pt"))

            # Initilize the new model that will be trained.
            model = VLE_model(gcn_saturation, mean_gcn, std_gcn, best_config, num_of_features)

            # Initialize the optimizer, learning rate scheduler and loss function.
            opt = Adam(model.parameters(), lr=best_config["learning_rate"])
            learning_rate_scheduler = ExponentialLR(opt, gamma=best_config["decay_rate"])
            loss_fn = torch.nn.MSELoss()

            # Stores the validation error for each epoch. 
            validation_error_for_each_epoch = []
            train_error_for_each_epoch = []

            # Code runs for "number_of_epochs" epochs.
            for epoch in range(number_of_epochs):   

                # Set the model to training mode
                model.train()
                train_loss = 0
                
                # For generating parity plots 
                train_predictions = []
                train_targets = []

                # Iterates over all the batches in dataset.
                for batch in train_loader:

                    # Unpack the batch to get the target values. 
                    (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

                    # Calculate the predictions for the given batch.
                    y_hat = model(batch)

                    # Check if there are NaN values in the output
                    if torch.isnan(y_hat).any():
                        nan_encountered = True
                        break

                    # Calculate the natural logarithm of the target values
                    ln_y_amine = torch.log(y_amine)
                    ln_y_water = torch.log(y_water)
                    sum_ln_y = ln_y_amine + ln_y_water

                    # Stack the targets 
                    targets = torch.stack((ln_y_amine, ln_y_water, sum_ln_y), dim=1)
                                
                    # Calculate the loss for the current batch
                    loss = loss_fn(y_hat, targets)
                    train_loss += loss.item()

                    # Apply backpropagation
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
                    train_predictions.extend(y_hat.detach().numpy())
                    train_targets.extend(targets.detach().numpy())
                if nan_encountered:
                    break
                train_loss /= len(train_loader)
                train_error_for_each_epoch.append(train_loss)

                # Update the learning rate after each epoch
                learning_rate_scheduler.step()

                # Evaluates how well the model does on the validation data
                model.eval() 
                validation_loss = 0
                validation_predictions = []
                validation_targets = []
                with torch.no_grad(): 

                    # Calculate the validation loss for all the batches in the validation dataset.
                    for batch in validation_loader:
                        # Unpack the batch to get the target values. 
                        (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

                        # Calculate the predictions for the given batch.
                        y_hat = model(batch)

                        # Check if there are NaN values in the output
                        if torch.isnan(y_hat).any():
                            nan_encountered = True
                            break

                        # Calculate the natural logarithm of the target values
                        ln_y_amine = torch.log(y_amine)
                        ln_y_water = torch.log(y_water)
                        sum_ln_y = ln_y_amine + ln_y_water

                        # Stack the targets 
                        targets = torch.stack((ln_y_amine, ln_y_water), dim=1)

                        # Calculate the validation loss for the current batch
                        validation_loss += loss_fn(y_hat, targets).item()

                        # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
                        validation_predictions.extend(y_hat.detach().numpy())
                        validation_targets.extend(targets.detach().numpy())

                if nan_encountered:
                    break

                # Calculate average validation loss (MSE) for this epoch based on the validation for each batch.
                validation_loss /= len(validation_loader)

                # Append the validation error for this epoch to the list.
                validation_error_for_each_epoch.append(validation_loss)

                # Want to print out how many percent left there is before the training for this split is done.
                print(f"Training for fold {index} is {round((epoch + 1) / number_of_epochs * 100, 2)}% done.")

                # Want to save parity plots for every 20th epoch, and the first epoch.
                if epoch % 10 == 0 or epoch == 0:
                    # Convert to tensors.
                    train_predictions = torch.tensor(np.stack(train_predictions), dtype=torch.float32)
                    train_targets = torch.tensor(np.stack(train_targets), dtype=torch.float32)
                    validation_predictions = torch.tensor(np.stack(validation_predictions), dtype=torch.float32)
                    validation_targets = torch.tensor(np.stack(validation_targets), dtype=torch.float32)

                    # Create a parity plot for the training and validation data.
                    cross_validation = True
                    create_parity_plot(train_predictions, train_targets, "Training", epoch, cross_validation, index)
                    create_parity_plot(validation_predictions, validation_targets, "Validation", epoch, cross_validation, index)

                """
                # If the validation loss is above 0.3 after 10 epochs, we want to reinitialize the model as the model is then stuck in a local minimum.
                if epoch==10 and validation_loss > 0.3 and times_initislized < 5:
                    initislize = True
                    times_initislized += 1
                    print("Reinitializing the model.")
                    break

                # If we are at the final epoch, we do not want to reinitialize the model.
                if epoch == number_of_epochs - 1:
                    initislize = False
                """

        # Validation error for each epoch gets stored in an Excel file.
        if index == 1:

            # Create a dataframe with the validation error for each epoch.
            df_validation = pd.DataFrame(validation_error_for_each_epoch, columns=[f"Fold {index}"])
            df_train = pd.DataFrame(train_error_for_each_epoch, columns=[f"Fold {index}"])

            # If the file already exists, I want to overwrite it.
            if os.path.exists(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx"):
                #input("Are you sure you want to delete the existing file? Press enter to continue.")
                os.remove(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx")

            # Have to make sure that both the folders exist. If it doesn't, create it.
            if not os.path.exists(f"Data/Combined model/GCN Cross Validation"):
                os.mkdir(f"Data/Combined model/GCN Cross Validation")

            # Save the dataframe to an Excel file.
            df_validation.to_excel(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx", index=False)
            df_train.to_excel(f"Data/Combined model/GCN Cross Validation/Train_error.xlsx", index=False)

        else:
            # Load the Excel file
            df_validation = pd.read_excel(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx")
            df_train = pd.read_excel(f"Data/Combined model/GCN Cross Validation/Train_error.xlsx")

            # Add the validation error for this fold to the dataframe
            df_validation[f"Fold {index}"] = validation_error_for_each_epoch
            df_train[f"Fold {index}"] = train_error_for_each_epoch

            # Save the dataframe to the Excel file.
            df_validation.to_excel(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx", index=False)
            df_train.to_excel(f"Data/Combined model/GCN Cross Validation/Train_error.xlsx", index=False)

        index += 1

    # Calculate the average validation error for each epoch.
    df_validation = pd.read_excel(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx")
    df_validation["Average"] = df_validation.mean(axis=1)
    df_validation.to_excel(f"Data/Combined model/GCN Cross Validation/Validation_error.xlsx", index=False)

    # Calculate the average training error for each epoch.
    df_train = pd.read_excel(f"Data/Combined model/GCN Cross Validation/Train_error.xlsx")
    df_train["Average"] = df_train.mean(axis=1)
    df_train.to_excel(f"Data/Combined model/GCN Cross Validation/Train_error.xlsx", index=False)

    plot_learning_curve()

#best_config = {'num_gcn_layers': 4, 'num_hidden_layers': 4, 'GCN_output_per_layer': [100, 420, 140, 140], 'hidden_neurons_per_layer': [260, 60, 180, 100], 'learning_rate': 0.007425096728429009, 'size_of_batch': 128, 'number_of_epochs': 1000, 'number_of_temp': 7, 'min_interval': 20, 'patience': 50, 'dropout_rate': 0.2,  'activation_function': 'relu', 'decay_rate': 0.95}
#train_with_CV(best_config, number_of_folds=2, number_of_epochs=100)


def split_on_amine(dev_dict, split_ratio=0.8):
    train_dict = {}
    test_dict = {}

    for cas_num, data in dev_dict.items():
        # Pick a random number between 0 and 1
        random_number = random.uniform(0, 1)
        if random_number < split_ratio: 
            train_dict[cas_num] = data

        else:
            test_dict[cas_num] = data

    return train_dict, test_dict


def evaluate_split(train_dict, test_dict):
    # Amount of data in the train data vs test data
    size_train = 0
    for key, data in train_dict.items():
        size_train += len(data["T"])

    size_test = 0
    for key, data in test_dict.items():
        size_test += len(data["T"])

    # Amount of amines in train data vs test data
    amine_ratio = len(train_dict)/(len(train_dict) + len(test_dict))
    data_ratio = size_train/(size_train + size_test)

    return amine_ratio, data_ratio


def train_model(best_config, number_of_epochs=300):
    """
    Main execution function.
    """

    # Create empty containors 
    split_ratios = [0.8]

    # Sets a seed for splitting data -> Found that 704 gives a good split for all ratios.
    seed_for_splitting = 704
    while True: 

        seed_beginning = seed_for_splitting

        for split_ratio in split_ratios:

            set_seed(seed_for_splitting)

            # Path to the folder where the code is located.
            path_to_code = os.path.dirname(os.path.realpath(__file__))

            # Extracts the mean and standard deviation for scaling. We use the same scaling as before. 
            mean_gcn, std_gcn = extract_mean_std(path_to_code, best_config["number_of_temp"], best_config["min_interval"])

            # Define the number of features for the GCN layer
            num_of_features = len(elements_allowed)

            # Extracts the development data as a dictionary. The dictionary contains the SMILES notation for all the compounds as keys,
            # and the corresponding VLE data as values. 
            dev_dict = extract_VLE_data("VLE_experimental_y_train.xlsx", path_to_code)

            # Want to split on amines to avoid data leakage between the training and validation data.
            train_data, validation_data = split_on_amine(dev_dict, split_ratio)

            amine_ratio, data_ratio = evaluate_split(train_data, validation_data)
            if round(amine_ratio, 1) ==  split_ratio and round(data_ratio, 1) == split_ratio:
                print(amine_ratio, data_ratio)
                pass

            else: 
                seed_for_splitting += 1
                break

        # If the seed has not changed during the while loop, we have found one that works.
        if seed_beginning == seed_for_splitting:
            break

    #split_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    split_ratios = [0.8]
    best_seed_for_each_ratio = []
    for split_ratio in split_ratios:

        set_seed(seed_for_splitting)

        # Path to the folder where the code is located.
        path_to_code = os.path.dirname(os.path.realpath(__file__))

        # Extracts the mean and standard deviation for scaling. We use the same scaling as before. 
        mean_gcn, std_gcn = extract_mean_std(path_to_code, best_config["number_of_temp"], best_config["min_interval"])

        # Define the number of features for the GCN layer
        num_of_features = len(elements_allowed)

        # Extracts the development data as a dictionary. The dictionary contains the SMILES notation for all the compounds as keys,
        # and the corresponding VLE data as values. 
        dev_dict = extract_VLE_data("VLE_experimental_y_train.xlsx", path_to_code)

        train_data, validation_data = split_on_amine(dev_dict, split_ratio)

        # Sets new seed for initializing the model.
        seeds = [1, 20, 31, 42, 49, 57, 68, 73, 89, 102]
        #seeds = [49]
        lowest_seed_error = 1000
        lowest_seed = None
        for seed in seeds: 
            # Sets a seed for for reinitializing the parameters in the current split. 
            nan_encountered = True
            while nan_encountered: 
                # If it does not converge, we will change seed in while loop. 
                set_seed(seed)
                nan_encountered = False
                print(seed)

                # Shuffle the training and validation data. Returns lists where the same index corresponds to the same molecule.
                shuffled_train_data = shuffle_data(train_data)
                shuffled_validation_data = shuffle_data(validation_data)

                # Convert smiles to graphs and prepare the data for the model.
                train_dataset_molecular = MolecularDataset(shuffled_train_data)
                validation_dataset_molecular = MolecularDataset(shuffled_validation_data)
                
                # Creating a DataLoader which handles batching and shuffling. collate_fn treat one batch of train_dataset
                # to get it on the right format for the GCN layer (converts graphs and adjacency matrices and maps them).
                train_loader = DataLoader(train_dataset_molecular, batch_size=best_config["size_of_batch"], collate_fn=merge_batch)
                validation_loader = DataLoader(validation_dataset_molecular, batch_size=best_config["size_of_batch"], collate_fn=merge_batch)

                # Initilize and load the best gcn model for predicting amine pressure from earlier training.
                gcn_saturation = GNN_Saturation_Pressure(best_config, num_of_features)
                gcn_saturation.load_state_dict(torch.load(f"Data/Best amine model/Best_model.pt"))

                # Initilize the new model that will be trained.
                model = VLE_model(gcn_saturation, mean_gcn, std_gcn, best_config, num_of_features)

                # Initialize the optimizer, learning rate scheduler and loss function.
                opt = Adam(model.parameters(), lr=best_config["learning_rate"])
                learning_rate_scheduler = ExponentialLR(opt, gamma=best_config["decay_rate"])
                loss_fn = torch.nn.MSELoss()

                # Stores the validation error for each epoch. 
                validation_error_for_each_epoch = []
                train_error_for_each_epoch = []

                # Code runs for "number_of_epochs" epochs.
                for epoch in range(number_of_epochs):   

                    # Set the model to training mode
                    model.train()
                    train_loss = 0
                    train_loss_without_sum = 0
                    
                    # For generating parity plots 
                    train_predictions = []
                    train_targets = []

                    # Iterates over all the batches in dataset.
                    for batch in train_loader:

                        # Unpack the batch to get the target values. 
                        (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

                        # Calculate the predictions for the given batch.
                        y_hat = model(batch)

                        # Check if there are NaN values in the output
                        if torch.isnan(y_hat).any():
                            nan_encountered = True
                            break

                        # Calculate the natural logarithm of the target values
                        ln_y_amine = torch.log(y_amine)
                        ln_y_water = torch.log(y_water)
                        sum_ln_y = ln_y_amine + ln_y_water

                        # Stack the targets 
                        targets = torch.stack((ln_y_amine, ln_y_water, sum_ln_y), dim=1)
                                    
                        # Calculate the loss for the current batch
                        loss = loss_fn(y_hat, targets)

                        # Apply backpropagation
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        # Loss without the sum
                        train_loss_without_sum += loss_fn(y_hat[:, :2], targets[:, :2]).item()

                        # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
                        train_predictions.extend(y_hat.detach().numpy())
                        train_targets.extend(targets.detach().numpy())
                    if nan_encountered:
                        break
                    train_loss_without_sum /= len(train_loader)
                    train_error_for_each_epoch.append(train_loss_without_sum)

                    # Update the learning rate after each epoch
                    learning_rate_scheduler.step()

                    # Evaluates how well the model does on the validation data
                    model.eval() 
                    validation_loss_without_sum = 0 
                    validation_predictions = []
                    validation_targets = []
                    with torch.no_grad(): 

                        # Calculate the validation loss for all the batches in the validation dataset.
                        for batch in validation_loader:
                            # Unpack the batch to get the target values. 
                            (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

                            # Calculate the predictions for the given batch.
                            y_hat = model(batch)

                            # Check if there are NaN values in the output
                            if torch.isnan(y_hat).any():
                                nan_encountered = True
                                break

                            # Calculate the natural logarithm of the target values
                            ln_y_amine = torch.log(y_amine)
                            ln_y_water = torch.log(y_water)
                            sum_ln_y = ln_y_amine + ln_y_water

                            # Stack the targets 
                            targets = torch.stack((ln_y_amine, ln_y_water, sum_ln_y), dim=1)

                            # Calculate the validation loss without the sum
                            validation_loss_without_sum += loss_fn(y_hat[:, :2], targets[:, :2]).item()

                            # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
                            validation_predictions.extend(y_hat.detach().numpy())
                            validation_targets.extend(targets.detach().numpy())

                    if nan_encountered:
                        break

                    # Calculate average validation loss (MSE) for this epoch based on the validation for each batch.
                    validation_loss_without_sum /= len(validation_loader)

                    # Append the validation error for this epoch to the list.
                    validation_error_for_each_epoch.append(validation_loss_without_sum)

                    # Want to save parity plots for every 10th epoch, and the first epoch.
                    if epoch % 10 == 0 or epoch == 0:
                        # Want to print out how many percent left there is before the training for this split is done.
                        #print(f"Training for is {round((epoch + 1) / number_of_epochs * 100, 2)}% done.")
                        print(f"Training for is {round((epoch) / number_of_epochs * 100, 2)}% done.")

                        # Convert to tensors.
                        train_predictions = torch.tensor(np.stack(train_predictions), dtype=torch.float32)
                        train_targets = torch.tensor(np.stack(train_targets), dtype=torch.float32)
                        validation_predictions = torch.tensor(np.stack(validation_predictions), dtype=torch.float32)
                        validation_targets = torch.tensor(np.stack(validation_targets), dtype=torch.float32)

                        # Create a parity plot for the training and validation data.
                        create_parity_plot(train_predictions, train_targets, "Training", epoch=epoch, seed=seed, split_ratio=split_ratio, cross_validation=False)
                        create_parity_plot(validation_predictions, validation_targets, "Validation", epoch=epoch, seed=seed, split_ratio=split_ratio, cross_validation=False)


                seed += 1


            seed -= 1

            # Create a dataframe with the validation error for each epoch.
            df_validation = pd.DataFrame(validation_error_for_each_epoch, columns=[f"Validation error"])
            df_train = pd.DataFrame(train_error_for_each_epoch, columns=[f"Train error"])

            # If the file already exists, I want to overwrite it.
            if os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Validation_error.xlsx"):
                #input("Are you sure you want to delete the existing file? Press enter to continue.")
                os.remove(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Validation_error.xlsx")

            # Have to make sure that both the folders exist. If it doesn't, create it.
            if not os.path.exists(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}"):
                os.mkdir(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}")

            # Save the dataframe to an Excel file.
            df_validation.to_excel(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Validation_error.xlsx", index=False)
            df_train.to_excel(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Train_error.xlsx", index=False)

            # Calculate the average validation error for each epoch.
            df_validation = pd.read_excel(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Validation_error.xlsx")
            df_validation["Average"] = df_validation.mean(axis=1)
            df_validation.to_excel(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Validation_error.xlsx", index=False)

            # Calculate the average training error for each epoch.
            df_train = pd.read_excel(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Train_error.xlsx")
            df_train["Average"] = df_train.mean(axis=1)
            df_train.to_excel(f"Data/Combined model/GCN models/Split ratio = {split_ratio}/{seed}/Train_error.xlsx", index=False)

            # As we use learning rate scheduler, the last validation error will be representative for the lowest (stable) validation error.
            if df_validation["Average"].iloc[-1] < lowest_seed_error:
                lowest_seed_error = df_validation["Average"].iloc[-1]
                lowest_seed = seed

            plot_learning_curve(cross_validation=False, seed=seed, split_ratio=split_ratio)
        
        print(f"Lowest error for split ratio {split_ratio} is seed {lowest_seed}: {lowest_seed_error}.")
        best_seed_for_each_ratio.append(lowest_seed)

    print(best_seed_for_each_ratio)

best_config_saturation_pressure_model = {'num_gcn_layers': 4, 'num_hidden_layers': 4, 'GCN_output_per_layer': [100, 420, 140, 140], 'hidden_neurons_per_layer': [260, 60, 180, 100], 'learning_rate': 0.007425096728429009, 'size_of_batch': 128, 'number_of_epochs': 1000, 'number_of_temp': 7, 'min_interval': 20, 'patience': 50, 'dropout_rate': 0.2,  'activation_function': 'relu', 'decay_rate': 0.95}
#train_model(best_config_saturation_pressure_model, number_of_epochs=100)


########################################################################################################################

# The part after this is for training and testing the final model. This is done after the optimal number of epochs has been found. 

########################################################################################################################



def train_final_model(best_config, number_of_epochs=300, load_model = False):
    """
    Main execution function for training the final model
    """


    # Path to the folder where the code is located.
    path_to_code = os.path.dirname(os.path.realpath(__file__))

    # Extracts the mean and standard deviation for scaling. We use the same scaling as before. 
    mean_gcn, std_gcn = extract_mean_std(path_to_code, best_config["number_of_temp"], best_config["min_interval"])

    # Define the number of features for the GCN layer
    num_of_features = len(elements_allowed)

    # Extracts the development data as a dictionary. The dictionary contains the SMILES notation for all the compounds as keys,
    # and the corresponding VLE data as values. 
    dev_dict = extract_VLE_data("VLE_experimental_y_train.xlsx", path_to_code)

    test_dict = extract_VLE_data("VLE_experimental_y_test.xlsx", path_to_code)

    # Sets a seed for initializing the model. 
    seed = 89

    # Sets the seed for initialization
    set_seed(seed)

    # Shuffle the training and validation data. Returns lists where the same index corresponds to the same molecule.
    shuffled_dev_data = shuffle_data(dev_dict)
    shuffled_test_data = shuffle_data(test_dict)

    # Convert smiles to graphs and prepare the data for the model.
    train_dataset_molecular = MolecularDataset(shuffled_dev_data)
    test_dataset_molecular = MolecularDataset(shuffled_test_data)
    
    # Creating a DataLoader which handles batching and shuffling. collate_fn treat one batch of train_dataset
    # to get it on the right format for the GCN layer (converts graphs and adjacency matrices and maps them).
    train_loader = DataLoader(train_dataset_molecular, batch_size=best_config["size_of_batch"], collate_fn=merge_batch)
    test_loader = DataLoader(test_dataset_molecular, batch_size=1, collate_fn=merge_batch)

    # Initilize and load the best gcn model for predicting amine pressure from earlier training.
    gcn_saturation = GNN_Saturation_Pressure(best_config, num_of_features)
    gcn_saturation.load_state_dict(torch.load(f"Data/Best amine model/Best_model.pt"))

    # Initilize the new model that will be trained.
    model = VLE_model(gcn_saturation, mean_gcn, std_gcn, best_config, num_of_features)

    # Initialize the optimizer, learning rate scheduler and loss function.
    opt = Adam(model.parameters(), lr=best_config["learning_rate"])
    learning_rate_scheduler = ExponentialLR(opt, gamma=best_config["decay_rate"])
    loss_fn = torch.nn.MSELoss()

    # If the model does not exist, we want to train it.
    if not load_model: 
        train_error_for_each_epoch = []
        # Train the model for "number_of_epochs" epochs.
        for epoch in range(number_of_epochs):   

            # Set the model to training mode
            model.train()
            train_loss_without_sum = 0

            # Iterates over all the batches in dataset.
            for batch in train_loader:

                # Unpack the batch to get the target values. 
                (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

                # Calculate the predictions for the given batch.
                y_hat = model(batch)

                # Calculate the natural logarithm of the target values
                ln_y_amine = torch.log(y_amine)
                ln_y_water = torch.log(y_water)
                sum_ln_y = ln_y_amine + ln_y_water

                # Stack the targets 
                targets = torch.stack((ln_y_amine, ln_y_water, sum_ln_y), dim=1)
                            
                # Calculate the loss for the current batch
                loss = loss_fn(y_hat, targets)

                # Apply backpropagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Loss without the sum
                train_loss_without_sum += loss_fn(y_hat[:, :2], targets[:, :2]).item()

            train_loss_without_sum /= len(train_loader)
            train_error_for_each_epoch.append(train_loss_without_sum)

            # Update the learning rate after each epoch
            learning_rate_scheduler.step()

            print(f"Training is {round((epoch) / number_of_epochs * 100, 2)}% done.")

        # Want to plot this to visualize the optimal number of epochs. x-axis is the epoch number and y-axis the validation error.
        plt.plot(train_error_for_each_epoch, label="Training Error")
        plt.xlabel("Epoch", fontsize=18)
        plt.ylabel("Train error [MSE]", fontsize=18)
        plt.title("Error for each epoch", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # Want to rotate the x-axis labels by 90 degrees so they don't overlap
        plt.xticks(rotation=45)
        # include grids for every 50 epochs
        plt.xticks(np.arange(50, len(train_error_for_each_epoch) + 50, 50))
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Data/Best combined model/Final evaluation/Learning_curve_train.png", dpi=300)
        plt.close()

        # Save this model as Data/Best combined model/Best_model.pt
        torch.save(model.state_dict(), f"Data/Best combined model/Best_model.pt")

    # Now want to evaluate how this model performs on the training data. 
    # Load the already existing model 
    model.load_state_dict(torch.load(f"Data/Best combined model/Best_model.pt"))
    
    # Want to evaluate how this model performs on the training data
    train_predictions = []
    train_targets = []
    train_loss_without_sum = 0

    # set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(train_loader):
            # Unpack the batch to get the target values. 
            (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

            # Calculate the predictions for the given batch.
            y_hat = model(batch)

            # Calculate the natural logarithm of the target values
            ln_y_amine = torch.log(y_amine)
            ln_y_water = torch.log(y_water)
            sum_ln_y = ln_y_amine + ln_y_water

            # Stack the targets 
            targets = torch.stack((ln_y_amine, ln_y_water, sum_ln_y), dim=1)

            # Calculate the validation loss for the current batch
            train_loss_without_sum += loss_fn(y_hat[:, :2], targets[:, :2]).item()

            # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
            train_predictions.extend(y_hat.detach().numpy())
            train_targets.extend(targets.detach().numpy())

        train_loss_without_sum /= len(train_loader)

        print(train_loss_without_sum)

        # Convert to tensors.
        train_predictions = torch.tensor(np.stack(train_predictions), dtype=torch.float32)
        train_targets = torch.tensor(np.stack(train_targets), dtype=torch.float32)

        # Saves the data to excel
        df_train = pd.DataFrame({"SMILES": shuffled_dev_data[0], "Temperature": shuffled_dev_data[5], "Predicted log(y_amine)": train_predictions[:, 0], "Actual log(y_amine)": train_targets[:, 0], "Predicted log(y_water)": train_predictions[:, 1], "Actual log(y_water)": train_targets[:, 1]})
        df_train.to_excel(f"Data/Best combined model/Final evaluation/Predicted_values_train.xlsx", index=False)

        # Create parity plot for the training data
        create_parity_plot(train_predictions, train_targets, "Train", final_model=True)

    # The model is now properly trained. Can now save it, and test it on the test data. 
    test_predictions = []
    test_targets = []
    test_loss_without_sum = 0

    # set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            # Unpack the batch to get the target values. 
            (x, edge_indices, batch_mapping), x_water, x_amine, y_water, y_amine, temperature, P_tot = batch

            # Calculate the predictions for the given batch.
            y_hat = model(batch)

            # Calculate the natural logarithm of the target values
            ln_y_amine = torch.log(y_amine)
            ln_y_water = torch.log(y_water)
            sum_ln_y = ln_y_amine + ln_y_water

            # Stack the targets 
            targets = torch.stack((ln_y_amine, ln_y_water, sum_ln_y), dim=1)

            # Calculate the validation loss for the current batch
            test_loss_without_sum += loss_fn(y_hat[:, :2], targets[:, :2]).item()

            # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
            test_predictions.extend(y_hat.detach().numpy())
            test_targets.extend(targets.detach().numpy())

        test_loss_without_sum /= len(test_loader)

        print(test_loss_without_sum)

        # Convert to tensors.
        test_predictions = torch.tensor(np.stack(test_predictions), dtype=torch.float32)
        test_targets = torch.tensor(np.stack(test_targets), dtype=torch.float32)

        # Saves the data to excel
        df_train = pd.DataFrame({"SMILES": shuffled_test_data[0], "Temperature": shuffled_test_data[5], "Predicted log(y_amine)": test_predictions[:, 0], "Actual log(y_amine)": test_targets[:, 0], "Predicted log(y_water)": test_predictions[:, 1], "Actual log(y_water)": test_targets[:, 1]})

        df_train.to_excel(f"Data/Best combined model/Final evaluation/Predicted_values_test.xlsx", index=False)

        # Create parity plot for the test data
        create_parity_plot(test_predictions, test_targets, "Test", final_model=True)


best_config = {'num_gcn_layers': 4, 'num_hidden_layers': 4, 'GCN_output_per_layer': [100, 420, 140, 140], 'hidden_neurons_per_layer': [260, 60, 180, 100], 'learning_rate': 0.007425096728429009, 'size_of_batch': 128, 'number_of_epochs': 1000, 'number_of_temp': 7, 'min_interval': 20, 'patience': 50, 'dropout_rate': 0.2,  'activation_function': 'relu', 'decay_rate': 0.95}
train_final_model(best_config, load_model=True)
