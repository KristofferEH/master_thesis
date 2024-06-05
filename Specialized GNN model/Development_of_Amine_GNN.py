# Imports all the necessary packages
import pandas as pd
import rdkit
import rdkit.Chem
import numpy as np
import random
import os

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

# only want to use 5 cpu cores. Using more will make the computer slow.
ray.init(num_cpus=5)

from collections import deque

# For creating the parity plot
import matplotlib.pyplot as plt

# My data only consists of 9 different elements. These are given in the dictionary below, along with their atom number.
elements_allowed = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def extract_training_SMILES_temperature_and_target(excel_filename, path_to_code):
    """
    Extracts the data that will be used for training and validating the model. This data is found in a folder called
    "T=temperature", where temperature is the temperature we want to focus on. The data is found in a file called
    "Development.xlsx" in this folder, which contains the SMILES notation for all the compounds, along with the
    corresponding vapor pressure at the given temperature. Will by default extract the data from the file
    "Development.xlsx".
    """

    # Locates the file for training data.
    excel_data = pd.read_excel(f"{path_to_code}/Data/{excel_filename}")

    X_dev = []
    y_dev = []
    temp = []    

    for _, row in excel_data.iterrows():
        SMILES = row['SMILES']
        for column in row.index:
            if column == 'SMILES':
                continue
            
            if column.startswith('T') and not column.startswith('TMIN') and not column.startswith('TMAX') and not column.startswith('Temperature Interval'):
                
                if not pd.isna(row[column]):  # Check if the value is not NaN
                    # Extracts the temperature and the logP value from the cells. 
                    cell_value = row[column].split("=")
                    temperature = float(cell_value[0][2:-2])
                    logP = float(cell_value[1])

                    # Appends the values to the lists.
                    X_dev.append(SMILES)
                    y_dev.append(logP)
                    temp.append(float(temperature))

    return X_dev, y_dev, temp


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


def shuffle_data(X_dev, y_dev, temp):
    """
    This function shuffles the data. 
    """

    # We have to shuffle the temperature and target values in the same way as the SMILES values.
    shuffled_indices = list(range(len(X_dev)))
    random.shuffle(shuffled_indices)

    X_dev = [X_dev[i] for i in shuffled_indices]
    y_dev = [y_dev[i] for i in shuffled_indices]
    temp = [temp[i] for i in shuffled_indices]

    return X_dev, y_dev, temp


def set_seed(seed_value):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)  # Set numpy seed
    torch.manual_seed(seed_value)  # Set torch seed
    random.seed(seed_value)  # Set python random seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Set python environment seed


def smiles2graph(sml):
    """
    This code is based on the code from the book "Deep Learning for Molecules and Materials" byAndrew D White.
    This function will return the graph of a molecule based on the SMILES string!
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


class MolecularDataset(Dataset):
    """
    This class is needed to create our dataset (on the Dataset format).
    The class inherits from the Dataset class.
    """

    def __init__(self, X, y, temperature):
        # Initializes the features and targets. Is out constructor.
        self.X = X
        self.y = y
        self.temperature = temperature

    def __len__(self):
        # Returns the length of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Extract the SMILES value of the molecule, calculate the graphs based on that SMILE, and then return this
        # value along with the corresponding target value.
        SMILES = self.X[idx]
        nodes, adj = smiles2graph(SMILES)

        # Convert nodes and adj to tensors, assuming they are NumPy arrays returned from smiles2graph
        nodes_tensor = torch.tensor(nodes, dtype=torch.float32)
        adj_tensor = torch.tensor(adj, dtype=torch.float32)

        # Convert the target value to a tensor. Assuming solubility is a single floating-point value.
        target_tensor = torch.tensor(self.y[idx], dtype=torch.float32)

        # Convert the temperature to a tensor. Assuming temperature is a single floating-point value.
        temperature_tensor = torch.tensor(self.temperature[idx], dtype=torch.float32)

        return (nodes_tensor, adj_tensor), target_tensor, temperature_tensor


def split_dataset(dataset, test_split=0.2):
    """
    Splits the dataset into training and testing datasets. Will split the dataset into 80% training and 20% testing
    if no test_split is specified.
    """
    # Determine the lengths
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size

    # Sets a seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Split the dataset. Implements a seed for reproducibility.
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    return train_dataset, test_dataset


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
    # Separate nodes, adjacency matrices, scalar tensors, and temperatures
    nodes_list = [item[0][0] for item in batch]
    adj_list = [item[0][1] for item in batch]
    scalar_tensors = [item[1] for item in batch]
    temperatures = [item[2] for item in batch]

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

    # Convert scalar_tensors (target values) into a single tensor (can be converted directly).
    merged_scalar_tensor = torch.stack(scalar_tensors)

    # Convert temperatures into a single tensor.
    merged_temperature_tensor = torch.stack(temperatures)

    return (merged_nodes_tensor, merged_edge_indices_tensor, batch_mapping_tensor), merged_scalar_tensor, merged_temperature_tensor


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


class GNNModel(nn.Module):
    """
    This class defines the structure of the model. The model will be a graph neural network (GNN) model. The model
    inherits from the nn.Module class, which is the base class for all neural network modules in PyTorch. 
    """

    def __init__(self, num_of_features, num_gcn_layers, num_hidden_layers, num_dense_neurons, GCN_output_per_layer, 
                 dropout_rate, activation_function):
        # Defines the structure of the model. 
        super().__init__()

        # Set the activation function.
        if activation_function == "relu":
            self.activation = nn.ReLU()

        elif activation_function == "sigmoid":
            self.activation = nn.Sigmoid()

        elif activation_function == "tanh":
            self.activation = nn.Tanh()

        # Initialize GCN layers and activations as ModuleList
        self.GCN_layers = nn.ModuleList()
        self.GCN_activations = nn.ModuleList()

        # Adds the GCN layers and the activation functions to the model.
        for i in range(num_gcn_layers):
            self.GCN_layers.append(GCNConv(num_of_features, GCN_output_per_layer[i]))
            self.GCN_activations.append(self.activation)
            num_of_features = GCN_output_per_layer[i]

        # Adds the global pooling layer.
        self.global_pooling = global_add_pool
        self.global_pooling_activation = self.activation

        # Initialize dense layers and activations as ModuleList
        self.dense_layers = nn.ModuleList()
        self.dense_activations = nn.ModuleList()
        self.dropout = nn.ModuleList()

        # Adds the dense layers and the activation functions to the model.
        for i in range(num_hidden_layers):
            self.dense_layers.append(nn.Linear(num_of_features, num_dense_neurons[i]))
            self.dense_activations.append(self.activation)
            self.dropout.append(nn.Dropout(p=dropout_rate))
            num_of_features = num_dense_neurons[i]

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



def train_model(config, path_to_code):
    """
    In this function, we set up a trainable function which will be called by Ray Tune. "config" is a
    dictionary containing the search space for all the hyperparameters.
    """
    
    # The hyperparameters are here extracted from the config argument.
    num_gcn_layers = config["num_gcn_layers"]
    num_hidden_layers = config["num_hidden_layers"]
    num_dense_neurons = config["hidden_neurons_per_layer"]
    GCN_output_per_layer = config["GCN_output_per_layer"]
    learning_rate = config["learning_rate"]
    size_of_batch = config["size_of_batch"]
    number_of_temp = config["number_of_temp"]
    min_interval = config["min_interval"]
    patience = config["patience"]
    dropout_rate = config["dropout_rate"]
    activation_function = config["activation_function"]
    decay_rate = config["decay_rate"]

    # Need to extract data from the correct file. We need SMILES, pressures and corresponding temperatures.
    X_dev, y_dev, temp = extract_training_SMILES_temperature_and_target("Train.xlsx", path_to_code, number_of_temp, min_interval)

    # Need to extract the mean and standard deviation for scaling. 
    mean, std = extract_mean_std(path_to_code, number_of_temp, min_interval)

    # Shuffle the data. Note that we have to shuffle the temperature and target values in the same way as the SMILES
    # values.
    X_dev, y_dev, temp = shuffle_data(X_dev, y_dev, temp)

    # Converting all molecules to graphs takes up too much memory. Makes a blueprint for the data instead.
    total_dataset = MolecularDataset(X_dev, y_dev, temp)

    # Split the dataset into train and test.
    train_dataset, test_dataset = split_dataset(total_dataset, test_split=0.3)

    # Creating a DataLoader which handles batching and shuffling. collate_fn treat the batch of train_dataset
    # to get it on the right format for the GCN layer.
    train_loader = DataLoader(train_dataset, batch_size=size_of_batch, collate_fn=merge_batch, shuffle=True)
    validation_loader = DataLoader(test_dataset, batch_size=size_of_batch, collate_fn=merge_batch, shuffle=True)

    # Will in our case be 9 as we only have 9 different atoms present.
    num_of_features = len(elements_allowed)

    # Creates the model, optimizer and loss function.
    model = GNNModel(num_of_features, num_gcn_layers, num_hidden_layers, num_dense_neurons, GCN_output_per_layer, 
                     dropout_rate, activation_function)

    opt = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Initilize learning rate decay 
    learning_rate_scheduler = ExponentialLR(opt, gamma=decay_rate)

    # Keeps track of the model states for the last "patience" epochs. This is used for early stopping.
    model_states_queue = deque(maxlen=(patience + 1))

    # Keep track of the best loss and the number of epochs without improvement
    best_loss = float("inf")
    epochs_without_improvement = 0

    # We run the code for "number_of_epochs" epochs.
    for epoch in range(config["number_of_epochs"]):
        torch.manual_seed(42)

        # Set the model to training mode
        model.train()  
        # Iterates over all the batches in the dataset.
        for batch in train_loader:
            (nodes, edge_indices, batch_mapping), targets, temperature = batch

            # Calculate the predictions. 
            y_hat = model(nodes, edge_indices,
                          batch_mapping, temperature, mean, std)
            
            # Scale the target values.
            targets = (targets - mean)/std

            # Calculate the loss for the current batch
            loss = loss_fn(y_hat, targets)

            # Apply backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Apply learning rate decay
        learning_rate_scheduler.step()
            
        # Evaluates how well the model does on the validation data
        model.eval() 
        validation_loss = 0
        with torch.no_grad(): 
            # Calculate the validation loss for all the batches in the validation dataset.
            for batch in validation_loader:
                (nodes, edge_indices, batch_mapping), targets, temperature = batch

                # Make predictions
                y_hat = model(nodes, edge_indices, batch_mapping, temperature, mean, std)

                # Scale the target values.
                targets = (targets - mean)/std

                # Calculate the validation loss for the current batch
                validation_loss += loss_fn(y_hat, targets).item()

        # Calculate average validation loss (MSE)
        validation_loss /= len(validation_loader)

        # Save the current model state in the queue
        model_states_queue.append({"epoch": epoch, "model_state": model.state_dict(), "loss": validation_loss})

        # Check for improvement
        if validation_loss < best_loss:
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping. If the validation loss has not improved for "patience" epochs, we break the loop.
        if epochs_without_improvement >= patience:
            # Retrieve the best model state, which is "patience" epochs before the last
            best_model_info = model_states_queue[0]

            # Load the best model state
            model.load_state_dict(best_model_info['model_state'])
            # Report the loss from the best model and break the loop.
            train.report({"loss": best_model_info['loss'], "best_epoch": best_model_info['epoch']})
            break

        # Report the validation loss if we didn't break the loop.
        train.report({"loss": validation_loss})


def create_parity_plot(y_hat_values, target_values, temperatures, seed, train_test_val, fold_number=None, epoch=None, cross_validation=False):
    # Plotting the parity line, that represents a perfect fit. To do this, we need to find the
    # maximum y value (can either be from the actual y value or the predicted value). We also need the minimum value.
    max_val = max(max(y_hat_values), max(target_values))
    min_val = min(min(y_hat_values), min(target_values))

    X_val_list = [min_val, max_val]
    y_val_list = [min_val, max_val]

    # Create a line between the minimum value and the maximum value. Note that here we have a list of X -values and lis
    plt.figure()
    # Want the x-axis and y-axis numbers to be larger so it becomes easier to read
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Want to rotate the x-axis labels by 90 degrees so they don't overlap
    plt.xticks(rotation=45)

    # Plots the actual values against the predicted values of the model.
    y_hat_40_to_70 = []
    target_40_to_70 = []
    y_hat_70_100 = []
    target_70_100 = []
    y_hat_100_135 = []
    target_100_135 = []

    for index, temp in enumerate(temperatures):
        if temp < 70:
            y_hat_40_to_70.append(y_hat_values[index])
            target_40_to_70.append(target_values[index])
        elif temp < 100:
            y_hat_70_100.append(y_hat_values[index])
            target_70_100.append(target_values[index])
        else:
            y_hat_100_135.append(y_hat_values[index])
            target_100_135.append(target_values[index])
    
    plt.plot(target_40_to_70, y_hat_40_to_70, "o", label="40°C-70°C", color="blue", markersize=3)
    plt.plot(target_70_100, y_hat_70_100, "o", label="70°C-100°C", color="orange", markersize=3)
    plt.plot(target_100_135, y_hat_100_135, "o", label="100°C-135°C", color="green", markersize=3)
    plt.plot(X_val_list, y_val_list, 'k--', label='Parity Line')
    plt.legend()

    # Creates title and labels
    plt.xlabel('Actual Values [log(mmHg)]', fontsize=18)
    plt.ylabel('Predicted Values [log(mmHg)]', fontsize=18)
    plt.title(f"Parity Plot - {train_test_val}", fontsize=18)

    # Saves the figure in a folder
    plt.tight_layout()
    if not cross_validation:
        plt.savefig(f"Data/Final model (amines)/{seed}/Parity Plot - {train_test_val}", dpi=300)
    else:
        if not os.path.exists(f"Data/GCN Cross Validation (amines)"):
            os.mkdir(f"Data/GCN Cross Validation (amines)")

        if not os.path.exists(f"Data/GCN Cross Validation (amines)/Parity plots"):
            os.mkdir(f"Data/GCN Cross Validation (amines)/Parity plots")

        if not os.path.exists(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}"):
            os.mkdir(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}")

        if not os.path.exists(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}/Epoch {epoch}"):
            os.mkdir(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}/Epoch {epoch}")

        plt.savefig(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}/Epoch {epoch}/Parity Plot - {train_test_val}", dpi=300)
    plt.close()

    # Also make three individual parity plots for each temperature range.
    plt.plot(target_40_to_70, y_hat_40_to_70, "o", label="40°C-70°C", color="blue", markersize=3)
    plt.plot(X_val_list, y_val_list, 'k--', label='Parity Line')
    plt.xlabel('Actual Values [log(mmHg)]', fontsize=18)
    plt.ylabel('Predicted Values [log(mmHg)]', fontsize=18)
    plt.title(f"Parity Plot - 40°C-70°C - {train_test_val}", fontsize=18)
    plt.legend()
    plt.tight_layout()

    if not cross_validation:
        plt.savefig(f"Data/Final model (amines)/{seed}/Parity Plot - 40°C-70°C - {train_test_val}", dpi=300)

    else: 
        plt.savefig(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}/Epoch {epoch}/Parity Plot - 40°C-70°C - {train_test_val}", dpi=300)

    plt.close()

    plt.figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45)
    plt.plot(target_70_100, y_hat_70_100, "o", label="70°C-100°C", color="orange", markersize=3)
    plt.plot(X_val_list, y_val_list, 'k--', label='Parity Line')
    plt.xlabel('Actual Values [log(mmHg)]', fontsize=18)
    plt.ylabel('Predicted Values [log(mmHg)]', fontsize=18)
    plt.title(f"Parity Plot - 70°C-100°C - {train_test_val}", fontsize=18)
    plt.legend()
    plt.tight_layout()
    if not cross_validation:
        plt.savefig(f"Data/Final model (amines)/{seed}/Parity Plot - 70°C-100°C - {train_test_val}", dpi=300)

    else:
        plt.savefig(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}/Epoch {epoch}/Parity Plot - 70°C-100°C - {train_test_val}", dpi=300)
    plt.close()

    plt.figure()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45)
    plt.plot(target_100_135, y_hat_100_135, "o", label="100°C-135°C", color="green", markersize=3)
    plt.plot(X_val_list, y_val_list, 'k--', label='Parity Line')
    plt.xlabel('Actual Values [log(mmHg)]', fontsize=18)
    plt.ylabel('Predicted Values [log(mmHg)]', fontsize=18)
    plt.title(f"Parity Plot - 100°C-135°C - {train_test_val}", fontsize=18)
    plt.legend()
    plt.tight_layout()
    if not cross_validation:
        plt.savefig(f"Data/Final model (amines)/{seed}/Parity Plot - 100°C-135°C - {train_test_val}", dpi=300)
    
    else:
        plt.savefig(f"Data/GCN Cross Validation (amines)/Parity plots/Fold {fold_number}/Epoch {epoch}/Parity Plot - 100°C-135°C - {train_test_val}", dpi=300)
    plt.close()



def train_model_with_CV(total_dataset, num_gcn_layers, num_hidden_layers, num_dense_neurons, GCN_output_per_layer,
                        learning_rate, size_of_batch, dropout_rate, mean, std, activation_function, decay_rate):
    """
    Cross-validation to determine the optimal number of epochs for training the best model.
    """

    number_of_epochs = 100

    # Want to implement cross validation. Will use 10-fold cross validation.
    number_of_folds = 10
    cross_validation = KFold(n_splits=number_of_folds, shuffle=True, random_state=42)

    index = 1
    for train_index, test_index in cross_validation.split(total_dataset):
        initislize = True
        times_initislized = 0

        # This is for reinitializing the model for a fold if it gets stuck in a local mimimum.
        while initislize:

            # Create subsets for training and testing based on the indices from the cross validation.
            train_dataset = Subset(total_dataset, train_index)
            test_dataset = Subset(total_dataset, test_index)

            # Creating a DataLoader, which handles batching and shuffling. The collate_fn will treat the batch of train_dataset
            # to make the data come on the right format for the GCN layer.
            train_loader = DataLoader(train_dataset, batch_size=size_of_batch, collate_fn=merge_batch, shuffle=True)
            validation_loader = DataLoader(test_dataset, batch_size=size_of_batch, collate_fn=merge_batch, shuffle=True)

            # Will in our case be 9, since we will only have 9 different atoms present.
            num_of_features = len(elements_allowed)

            # Creates the model, optimizer and loss function.
            model = GNNModel(num_of_features, num_gcn_layers, num_hidden_layers, num_dense_neurons, GCN_output_per_layer, dropout_rate, activation_function)
            opt = Adam(model.parameters(), lr=learning_rate)
            loss_fn = torch.nn.MSELoss()

            # Initilize learning rate decay
            learning_rate_scheduler = ExponentialLR(opt, gamma=decay_rate)

            # Loads the best general model 
            model.load_state_dict(torch.load(f"Data/Best general model/Best_model.pt"))

            # Stores the validation error for each epoch. 
            validation_error_for_each_epoch = []
            train_error_for_each_epoch = []

            # Code runs for "number_of_epochs" epochs.
            for epoch in range(number_of_epochs):

                # Want to reinitialize the model if the validation error is above 0.35 after 10 epochs, but a maximum of 5 times.
                # Changing seed will both shuffle the loaders and initialize weights of the model again.
                seeds = [42, 20, 30, 15, 50]
                torch.manual_seed(seeds[times_initislized])
      
                # Set the model to training mode
                model.train()
                train_loss = 0
                train_predictions = []
                train_targets = []
                temperature_train = []

                # Iterates over all the batches in dataset.
                for batch in train_loader:
                    # Extract the data from the batch
                    (nodes, edge_indices, batch_mapping), targets, temperature = batch

                    # Calculate the predictions.
                    y_hat = model(nodes, edge_indices, batch_mapping, temperature, mean, std)
                    
                    # Scale the target values.
                    targets = (targets - mean)/std
                                
                    # Calculate the loss for the current batch
                    loss = loss_fn(y_hat, targets)
                    train_loss += loss.item()

                    # Apply backpropagation
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # Add the predictions and targets to the lists for the parity plots. Note that
                    # the parity plots are in log(mmHg), so we need to convert the values back to the original scale.
                    y_hat = y_hat * std + mean
                    targets = targets * std + mean

                    # Add the predictions and targets to the lists. Dont want to append the tensor, but the values.
                    train_predictions.extend(y_hat.detach().numpy())
                    train_targets.extend(targets.detach().numpy())
                    temperature_train.extend(temperature.detach().numpy())
                
                train_loss /= len(train_loader)
                train_error_for_each_epoch.append(train_loss)

                # Apply learning rate decay
                learning_rate_scheduler.step()

                # Evaluates how well the model does on the validation data
                model.eval() 
                validation_loss = 0
                validation_predictions = []
                validation_targets = []
                validation_temperatures = []
                with torch.no_grad(): 

                    # Calculate the validation loss for all the batches in the validation dataset.
                    for batch in validation_loader:
                        # Extract the data from the batch
                        (nodes, edge_indices, batch_mapping), targets, temperature = batch

                        # Make predictions
                        y_hat = model(nodes, edge_indices, batch_mapping, temperature, mean, std)

                        # Scale the target values.
                        targets = (targets - mean)/std

                        # Calculate the validation loss for the current batch
                        validation_loss += loss_fn(y_hat, targets).item()

                        # Add the predictions and targets to the lists for the parity plots. Note that
                        # the parity plots are in log(mmHg), so we need to convert the values back to the original scale.
                        y_hat = y_hat * std + mean
                        targets = targets * std + mean
                        validation_predictions.extend(y_hat.detach().numpy())
                        validation_targets.extend(targets.detach().numpy())
                        validation_temperatures.extend(temperature.detach().numpy())

                # Calculate average validation loss (MSE) for this epoch based on the validation for each batch.
                validation_loss /= len(validation_loader)

                # Append the validation error for this epoch to the list.
                validation_error_for_each_epoch.append(validation_loss)

                # Want to save parity plots for every 20th epoch, and the first epoch.
                if epoch % 10 == 0 or epoch == 0:
                    # Create a parity plot for the training and validation data.
                    create_parity_plot(train_predictions, train_targets, temperature_train, seeds[times_initislized], "Training", index, epoch, cross_validation=True)
                    create_parity_plot(validation_predictions, validation_targets, validation_temperatures, seeds[times_initislized], "Validation", index, epoch, cross_validation=True)


                # Want to print out how many percent left there is before the training for this split is done.
                print(f"Training for fold {index} is {round((epoch + 1) / number_of_epochs * 100, 2)}% done.")

                # If the validation loss is above 0.3 after 10 epochs, we want to reinitialize the model as the model is then stuck in a local minimum.
                if epoch==10 and validation_loss > 0.3 and times_initislized < 5:
                    initislize = True
                    times_initislized += 1
                    print("Reinitializing the model.")
                    break

                # If we are at the final epoch, we do not want to reinitialize the model.
                if epoch == number_of_epochs - 1:
                    initislize = False


        # Validation error for each epoch gets stored in an Excel file.
        if index == 1:

            # Create a dataframe with the validation error for each epoch.
            df_validation = pd.DataFrame(validation_error_for_each_epoch, columns=[f"Fold {index}"])
            df_train = pd.DataFrame(train_error_for_each_epoch, columns=[f"Fold {index}"])

            # If the file already exists, I want to overwrite it.
            if os.path.exists(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx"):
                input("Are you sure you want to delete the existing file? Press enter to continue.")
                os.remove(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx")

            # Have to make sure that both the folders exist. If it doesn't, create it.
            if not os.path.exists(f"Data/GCN Cross Validation (amines)"):
                os.mkdir(f"Data/GCN Cross Validation (amines)")

            # Save the dataframe to an Excel file.
            df_validation.to_excel(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx", index=False)
            df_train.to_excel(f"Data/GCN Cross Validation (amines)/Train_error.xlsx", index=False)

        else:
            # Load the Excel file
            df_validation = pd.read_excel(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx")
            df_train = pd.read_excel(f"Data/GCN Cross Validation (amines)/Train_error.xlsx")

            # Add the validation error for this fold to the dataframe
            df_validation[f"Fold {index}"] = validation_error_for_each_epoch
            df_train[f"Fold {index}"] = train_error_for_each_epoch

            # Save the dataframe to the Excel file.
            df_validation.to_excel(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx", index=False)
            df_train.to_excel(f"Data/GCN Cross Validation (amines)/Train_error.xlsx", index=False)

        index += 1

    # Calculate the average validation error for each epoch.
    df_validation = pd.read_excel(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx")
    df_validation["Average"] = df_validation.mean(axis=1)
    df_validation.to_excel(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx", index=False)

    # Calculate the average training error for each epoch.
    df_train = pd.read_excel(f"Data/GCN Cross Validation (amines)/Train_error.xlsx")
    df_train["Average"] = df_train.mean(axis=1)
    df_train.to_excel(f"Data/GCN Cross Validation (amines)/Train_error.xlsx", index=False)


def train_with_CV(best_config):
    """
    Main execution function for training the model with cross validation.
    """

    # Sets a seed for reproducibility
    set_seed(42)

    # Extracts the best hyperparameters from the file.
    number_of_temperatures = best_config["number_of_temp"]
    min_interval = best_config["min_interval"]

    # Path to the folder where the code is located.
    path_to_code = os.path.dirname(os.path.realpath(__file__))

    # Extracts the amine data. 
    X_dev, y_dev, temp = extract_training_SMILES_temperature_and_target("Amines_train.xlsx", path_to_code)

    # Need to extract the mean and standard deviation for scaling. We use the same scaling as before. 
    mean, std = extract_mean_std(path_to_code, number_of_temperatures, min_interval)

    # Shuffle the data. Note that we have to shuffle the temperature and target values in the same way as the SMILES
    # values.
    X_dev, y_dev, temp = shuffle_data(X_dev, y_dev, temp)

    # Converting all molecules to graphs takes up too much memory. Makes a blueprint for the data instead.
    total_dataset = MolecularDataset(X_dev, y_dev, temp)

    # Specify the hyperparameters for the chosen model that you would like to perform cross validation on.
    num_gcn_layers = best_config["num_gcn_layers"]
    num_hidden_layers = best_config["num_hidden_layers"]
    num_dense_neurons = best_config["hidden_neurons_per_layer"]
    GCN_output_per_layer = best_config["GCN_output_per_layer"]
    learning_rate = best_config["learning_rate"]
    size_of_batch = best_config["size_of_batch"]
    dropout_rate = best_config["dropout_rate"]
    activation_function = best_config["activation_function"]
    decay_rate = best_config["decay_rate"]

    # Will now train a model with the hyperparameters set inside the function using cross validation.
    train_model_with_CV(total_dataset, num_gcn_layers, num_hidden_layers, num_dense_neurons, GCN_output_per_layer,
                        learning_rate, size_of_batch, dropout_rate, mean, std, activation_function, decay_rate)

    # Want to extract the average data created by "train_model_with_CV" from the Excel file. -> This was used to find the 
    # lowest validation error. 
    df_validation = pd.read_excel(f"Data/GCN Cross Validation (amines)/Validation_error.xlsx")
    average_validation_error = df_validation["Average"].tolist()

    # Save the training error as well.
    df_train = pd.read_excel(f"Data/GCN Cross Validation (amines)/Train_error.xlsx")
    average_train_error = df_train["Average"].tolist()

    # Print out the lowest validation error
    lowest_error = min(average_validation_error)
    print(lowest_error)

    # Save the lowest error in a text file.
    with open(f"Data/GCN Cross Validation (amines)/Lowest_error.txt", "w") as file:
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
    plt.savefig(f"Data/GCN Cross Validation (amines)/Validation_error_plot.png", dpi=300)

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
    plt.ylabel("Cross-validation error [MSE]", fontsize=18)
    plt.title("Error for each epoch", fontsize=18)
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
    plt.savefig(f"Data/GCN Cross Validation (amines)/Validation_error_plot_excluding_first_100.png", dpi=300)

best_config = {'num_gcn_layers': 4, 'num_hidden_layers': 4, 'GCN_output_per_layer': [100, 420, 140, 140], 'hidden_neurons_per_layer': [260, 60, 180, 100], 'learning_rate': 0.007425096728429009, 'size_of_batch': 128, 'number_of_epochs': 1000, 'number_of_temp': 7, 'min_interval': 20, 'patience': 50, 'dropout_rate': 0.2,  'activation_function': 'relu', 'decay_rate': 0.95}
#train_with_CV(best_config)





########################################################################################################################

# The part after this is for training and testing the final amine model. This is done after the optimal number of epochs has been found
# using cross validation on the best general model.

########################################################################################################################





def train_final_model(total_dataset, test_dataset, num_gcn_layers, num_hidden_layers, num_dense_neurons,
                      GCN_output_per_layer, learning_rate, size_of_batch, number_of_epochs, dropout_rate, smiles_train, smiles_test,
                      mean, std, activation_function, seed, decay_rate, load_model=False):
    """
    This trains the final model and tests it against the test data. Marks the end of the project.
    """

    # Will in our case be 9, since we will only have 9 different atoms present.
    num_of_features = len(elements_allowed)

    # Creates the model, optimizer and loss function.
    model = GNNModel(num_of_features, num_gcn_layers, num_hidden_layers, num_dense_neurons, GCN_output_per_layer, 
                     dropout_rate, activation_function)
    opt = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    # Initilize learning rate decay
    learning_rate_scheduler = ExponentialLR(opt, gamma=decay_rate)

    total_dataset_not_shuffled = total_dataset

    # Creating a DataLoader, which handles batching and shuffling.
    train_loader = DataLoader(total_dataset, batch_size=size_of_batch, collate_fn=merge_batch, shuffle=True)

    # Want to test one and one molecule, so we set the batch size to 1.
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=merge_batch, shuffle=False)

    if not load_model:
        # Loads the best general model 
        model.load_state_dict(torch.load(f"Data/Best general model/Best_model.pt"))

        train_error_for_each_epoch = []
        model.train()
        # We run the code for "number_of_epochs" epochs.
        for epoch in range(number_of_epochs):
            torch.manual_seed(seed)
            train_loss = 0
            # Iterates over all the batches in the dataset.
            for batch in train_loader:
                (nodes, edge_indices, batch_mapping), targets, temperature = batch
                y_hat = model(nodes, edge_indices, batch_mapping, temperature, mean, std).squeeze()

                # Scale the target values.
                targets = (targets - mean)/std

                loss = loss_fn(y_hat, targets)

                train_loss += loss.item()

                # Apply backpropagation
                opt.zero_grad()
                loss.backward()
                opt.step()

            train_loss /= len(train_loader)
            train_error_for_each_epoch.append(train_loss)

            # Want to print out how many percent left there is before the training for this split is done.
            print(f"Training is {round((epoch + 1) / number_of_epochs * 100, 2)}% done.")

            # Apply learning rate decay
            learning_rate_scheduler.step()

            # Create a folder for the final model if it doesn't exist.
            if not os.path.exists(f"Data/Final model (amines)"):
                os.mkdir(f"Data/Final model (amines)")

            # Create a folder with the seed number if it doesn't exist.
            if not os.path.exists(f"Data/Final model (amines)/{seed}"):
                os.mkdir(f"Data/Final model (amines)/{seed}")

            # Want to save the model
            torch.save(model.state_dict(), f"Data/Final model (amines)/{seed}/Best_model.pt")

            # Plot the training error for each epoch.
            plt.plot(train_error_for_each_epoch, label="Training Error")
            plt.xlabel("Epoch", fontsize=18)
            plt.ylabel("Training error [MSE]", fontsize=18)
            plt.title("Training error for each epoch", fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # Want to rotate the x-axis labels by 90 degrees so they don't overlap
            plt.xticks(rotation=45)
            # include grids for every 50 epochs
            plt.xticks(np.arange(50, number_of_epochs + 50, 50))
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"Data/Final model (amines)/{seed}/Training_error_plot.png", dpi=300)
            plt.close()

    else: 
        # Load the model
        model.load_state_dict(torch.load(f"Data/Final model (amines)/{seed}/Best_model.pt"))

    # Set the model to evaluation mode
    model.eval()  

    # Want to keep track of how well the model does on training data
    train_error = 0
    y_hat_values_train = []
    target_values_train = []
    temperature_train = []

    # Have to create a new DataLoader where we don't shuffle the data so that it is possible to match it with the SMILES
    train_loader = DataLoader(total_dataset_not_shuffled, batch_size=1, collate_fn=merge_batch, shuffle=False)

    # Want to store the predictions of the training data in a dataframe.
    df_train = pd.DataFrame(columns=["SMILES", "Temperature", "Predicted log(P)", "Actual log(P)"])

    with torch.no_grad():  # No gradients needed for testing
        # Calculate the train loss for all the batches in the train dataset.
        for index, batch in enumerate(train_loader):
            (nodes, edge_indices, batch_mapping), targets, temperature = batch
            y_hat = model(nodes, edge_indices, batch_mapping, temperature, mean, std).squeeze()

            # Want to scale the predictions back to log(mmHg)
            y_hat = y_hat * std + mean

            # Add this to the dataframe
            df_train.at[index, "SMILES"] = smiles_train[index]
            df_train.at[index, "Temperature"] = temperature.item()
            df_train.at[index, "Predicted log(P)"] = y_hat.item()
            df_train.at[index, "Actual log(P)"] = targets.item()

            # Append the values to the lists. 
            y_hat_values_train.append(y_hat.item())
            target_values_train.append(targets.item())
            temperature_train.append(temperature)

            train_error += loss_fn((y_hat).unsqueeze(0), targets).item()

    train_error /= len(train_loader)

    # Sort the dataframe based on the SMILES and temperature values.
    df_train = df_train.sort_values(by=["SMILES", "Temperature"])

    # Save the dataframe to an Excel file.
    df_train.to_excel(f"Data/Final model (amines)/{seed}/Predicted_values_train.xlsx", index=False)

    # Set the model to evaluation mode
    model.eval()  

    # Want to store the predictions in a dataframe.
    df = pd.DataFrame(columns=["SMILES", "Temperature", "Predicted log(P)", "Actual log(P)"])

    # We now want to calculate how well this model does on the test data
    test_error = 0
    y_hat_values_test = []
    target_values_test = []
    temperature_test = []
    with torch.no_grad():  # No gradients needed for testing
        # Calculate the test loss for all the batches in the test dataset.
        for index, batch in enumerate(test_loader):
            (nodes, edge_indices, batch_mapping), targets, temperature = batch
            y_hat = model(nodes, edge_indices, batch_mapping, temperature, mean, std).squeeze()

            # Want to scale the predictions back to log(mmHg)
            y_hat = y_hat * std + mean

            # Add this to the dataframe
            df.at[index, "SMILES"] = smiles_test[index]
            df.at[index, "Temperature"] = temperature.item()
            df.at[index, "Predicted log(P)"] = y_hat.item()
            df.at[index, "Actual log(P)"] = targets.item()

            # Append the values to the lists.
            y_hat_values_test.append(y_hat.item())
            target_values_test.append(targets.item())
            temperature_test.append(temperature.item())
            # Have to unsqeeze the y_hat value, as loss_fn wants the tensor values to be in a list. When you have a
            # batch size of 1 and squeeze, you just get a scaler instead of a 1D list. This is why we have to
            # unsqueeze it again. Note that loss_fn will give the MSE for each batch
            test_error += loss_fn((y_hat).unsqueeze(0), targets).item()

    # Calculate average test error (MSE)
    test_error /= len(test_loader)

    # Sort the dataframe based on the SMILES and temperature values.
    df = df.sort_values(by=["SMILES", "Temperature"])

    # Save the dataframe to an Excel file.
    df.to_excel(f"Data/Final model (amines)/{seed}/Predicted_values_test.xlsx", index=False)

    # Print out the test error as MSE
    print("Note that the following errors should not be the same as the ones in the Excel file, as these values are scaled back.")
    print(f"\nThe training error (MSE) is {train_error} log(mmHg)^2.")
    print(f"The test error (MSE) is {test_error} log(mmHg)^2.")

    # Also want to save this to a text file.
    with open(f"Data/Final model (amines)/{seed}/Errors.txt", "w") as file:
        file.write(f"The training error (MSE) is {train_error} log(mmHg)^2.")
        file.write("\n")
        file.write(f"The test error (MSE) is {test_error} log(mmHg)^2.")

    # Want to save the model
    torch.save(model.state_dict(), f"Data/Final model (amines)/{seed}/Best_model.pt")

    # Create a parity plot for the training and test data.
    create_parity_plot(y_hat_values_train, target_values_train, temperature_train, seed, "Train")
    create_parity_plot(y_hat_values_test, target_values_test, temperature_test, seed, "Test")

    # Want to convert the y_values from log(mmHg) to mmHg. 
    y_hat_values_train = [10 ** i for i in y_hat_values_train]
    target_values_train = [10 ** i for i in target_values_train]
    y_hat_values_test = [10 ** i for i in y_hat_values_test]
    target_values_test = [10 ** i for i in target_values_test]


    # Calculate and print the training error in mmHg
    train_error = 0
    for i in range(len(y_hat_values_train)):
        train_error += (y_hat_values_train[i] - target_values_train[i]) **2
    train_error = np.sqrt(train_error / len(y_hat_values_train))
    print(f"\nThe training error (RMSE) is {train_error} mmHg.")

    # Calculate and print the test error in Pa
    test_error = 0
    for i in range(len(y_hat_values_test)):
        test_error += (y_hat_values_test[i] - target_values_test[i]) ** 2
    test_error = np.sqrt(test_error / len(y_hat_values_test))
    print(f"The test error (RMSE) is {test_error} mmHg.")

    # Also want to save this to a text file.
    with open(f"Data/Final model (amines)/{seed}/Errors.txt", "a") as file:
        file.write(f"\nThe training error (RMSE) is {train_error} mmHg.")
        file.write("\n")
        file.write(f"The test error (RMSE) is {test_error} mmHg.")

@ray.remote
def final_model(best_config, seed):
    """
    Main execution function. For now, we only focus on one temperature.
    """

    # Sets a seed for reproducibility
    set_seed(seed)

    # Path to the folder where the code is located.
    path_to_code = os.path.dirname(os.path.realpath(__file__))

    # Extracts the number of temperatures and the minimum interval from the best_config dictionary.
    number_of_temperatures = best_config["number_of_temp"]
    min_interval = best_config["min_interval"]

    # Need to extract the data from the file. When this data was made the data was shuffled. 
    X_dev, y_dev, temp_dev = extract_training_SMILES_temperature_and_target("Amines_train.xlsx", path_to_code)
    X_test, y_test, temp_test = extract_training_SMILES_temperature_and_target("Amines_test.xlsx", path_to_code)

    # Need to extract the mean and standard deviation for scaling.
    mean, std = extract_mean_std(path_to_code, number_of_temperatures, min_interval)

    # Shuffles data
    X_dev, y_dev, temp_dev = shuffle_data(X_dev, y_dev, temp_dev)
    X_test, y_test, temp_test = shuffle_data(X_test, y_test, temp_test)

    # Convert the data to a blueprint.
    total_dataset = MolecularDataset(X_dev, y_dev, temp_dev)
    test_dataset = MolecularDataset(X_test, y_test, temp_test)

    # Extract the optimal parameters from the text file.
    num_gcn_layers = best_config["num_gcn_layers"]
    num_hidden_layers = best_config["num_hidden_layers"]
    num_dense_neurons = best_config["hidden_neurons_per_layer"]
    GCN_output_per_layer = best_config["GCN_output_per_layer"]
    learning_rate = best_config["learning_rate"]
    size_of_batch = best_config["size_of_batch"]
    dropout_rate = best_config["dropout_rate"]
    activation_function = best_config["activation_function"]
    decay_rate = best_config["decay_rate"]


    # Input the number of epochs manually based on the CV plot.
    number_of_epochs = 100

    # Train the final model. 
    train_final_model(total_dataset, test_dataset, num_gcn_layers, num_hidden_layers, num_dense_neurons,
                      GCN_output_per_layer, learning_rate, size_of_batch, number_of_epochs, dropout_rate, X_dev, X_test,
                      mean, std, activation_function, seed, decay_rate, load_model=False)


best_config = {'num_gcn_layers': 4, 'num_hidden_layers': 4, 'GCN_output_per_layer': [100, 420, 140, 140], 'hidden_neurons_per_layer': [260, 60, 180, 100], 'learning_rate': 0.007425096728429009, 'size_of_batch': 128, 'number_of_epochs': 1000, 'number_of_temp': 7, 'min_interval': 20, 'patience': 50, 'dropout_rate': 0.2,  'activation_function': 'relu', 'decay_rate': 0.95}
# 15, 42, 84, 19, 37, 45, 87, 57, 61, 1, 3, 7
seeds = [7, 84, 61, 42, 15]

objects = []
for seed in seeds:
    objects.append(final_model.remote(best_config, seed))

results = ray.get(objects)

"""
# Iterate over all the seeds and extract the errors for each seed.
error_strings = []
for seed in seeds:
    with open(f"Data/Final model/{seed}/Errors.txt", "r") as file:
        error_strings.append(file.read())

error_values = []
for error in error_strings:
    if "The test error (MSE) is" in error: 
        start = error.find("The test error (MSE) is") + len("The test error (MSE) is ")
        end = error.find(" log(mmHg)^2", start)
        error_values.append(float(error[start:end]))

# Plot the test errors for each seed.
plt.plot(seeds, error_values, "o")
plt.xlabel("Seed", fontsize=18)
plt.ylabel("Test error (MSE) [log(mmHg)^2]", fontsize=18)
plt.title("Test error for each seed", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the figure
plt.tight_layout()
plt.show()
"""

#### This is for evaluating the model on specific molecules. ###
"""
df_pred_test = pd.read_excel(f"Data/Final model/Predicted_values_test.xlsx")
df_pred_train = pd.read_excel(f"Data/Final model/Predicted_values_train.xlsx")
df_train = pd.read_excel(f"Data/Train.xlsx")
df_test = pd.read_excel(f"Data/Test.xlsx")

#df, df_pred = df_train, df_pred_train
df, df_pred = df_test, df_pred_test

change_in_pressure_list = []
change_in_temperature_list = []

for index, smiles in enumerate(df_pred["SMILES"].unique()):
    # The predicted data
    df_temp = df_pred[df_pred["SMILES"] == smiles]
    # For extracting the antonine data
    df_test_temp = df[df["SMILES"] == smiles]

    # Get the lowest and highest temperature for the molecule.
    min_temp = df_temp["Temperature"].min()
    max_temp = df_temp["Temperature"].max()

    # Generate a temperature range for the molecule.
    temperature_range = np.arange(min_temp, max_temp + 1, 1)

    # Calculate the pressure for the temperature range.
    if df_test_temp.shape[0] > 1:
        print(f"Duplicate: {smiles}")
        continue
    A = df_test_temp["A"].item()
    B = df_test_temp["B"].item()
    C = df_test_temp["C"].item()
    pressure = []
    for temp in temperature_range:
        # Here, P will be the pressure in mmHg! 
        log_P = A - B / (temp + C)
        pressure.append(log_P)

    change_in_pressure_list.append(df_temp["Predicted log(P)"].iloc[-1] - df_temp["Predicted log(P)"].iloc[0])
    change_in_temperature_list.append(max_temp - min_temp)

    # Find the rate of increase for the predicted values. 
    rate_of_increase_pred = (df_temp["Predicted log(P)"].iloc[-1] - df_temp["Predicted log(P)"].iloc[0]) / (max_temp - min_temp)

    # Find the rate of increase for the Antonine values.
    rate_of_increase_antonine = (pressure[-1] - pressure[0]) / (max_temp - min_temp)

    # Add a new column "Squared Difference"
    df_pred["MSE"] = (df_pred["Predicted log(P)"] - df_pred["Actual log(P)"])**2

    # Take the average of the MSE for each molecule.
    average_MSE = df_pred.groupby("SMILES")["MSE"].mean()

    # Sort average_MSE in descending order. 
    average_MSE = average_MSE.sort_values(ascending=False)

    # For finding linear trends. 
    #if df_temp["Predicted log(P)"].iloc[-1] - df_temp["Predicted log(P)"].iloc[0] < 0.2:

    # For predicting the copmounds with the worst rate of increase fit. 
    #if abs(rate_of_increase_pred - rate_of_increase_antonine) > 0.016:

    # For predicting the compounds with the worst MSE.
    if smiles in average_MSE.index[:10]:

        plt.plot(temperature_range, pressure, label="Antonine")
        plt.plot(df_temp["Temperature"], df_temp["Predicted log(P)"], "o", label="Predicted")
        plt.xlabel("Temperature [°C]", fontsize=18)
        plt.ylabel("log(P) [mmHg]", fontsize=18)
        plt.title(f"Pressure for {smiles}", fontsize=18)
        plt.legend()
        plt.tight_layout()

        # If there is no folder called "Evaluation on molecules", create it.
        if not os.path.exists(f"Data/Final model/Evaluation on molecules"):
            os.mkdir(f"Data/Final model/Evaluation on molecules")
        
        plt.savefig(f"Data/Final model/Evaluation on molecules/Pressure for {smiles}.png", dpi=300)
        plt.close()

print("DONE")
"""


'''
    smiles_with_low_change = ["C(CCCCF)CCCCCl", "C=CC1=C(C=CC=C1Cl)Cl", "C1CCCCCSCCCC1"]
    smiles_with_high_change = ["C(=O)(OS(=O)(=O)F)F", "C(C(C(F)(F)F)(F)F)(Cl)Cl", "C(C(CF)(F)Cl)F"]

    if smiles in smiles_with_low_change:

        plt.plot(temperature_range, pressure, label="Antonine")
        plt.plot(df_temp["Temperature"], df_temp["Predicted log(P)"], "o", label="Predicted")
        plt.xlabel("Temperature [°C]", fontsize=18)
        plt.ylabel("log(P) [mmHg]", fontsize=18)
        plt.title(f"Pressure for {smiles}", fontsize=18)
        plt.legend()
        plt.tight_layout()
        # Create folder evaluation on molecules if it doesn't exist.
        if not os.path.exists(f"Data/Final model/Evaluation on molecules"):
            os.mkdir(f"Data/Final model/Evaluation on molecules")
        plt.savefig(f"Data/Final model/Evaluation on molecules/Pressure for {smiles}.png", dpi=300)
        plt.close()

    if smiles in smiles_with_high_change:

        plt.plot(temperature_range, pressure, label="Antonine")
        plt.plot(df_temp["Temperature"], df_temp["Predicted log(P)"], "o", label="Predicted")
        plt.xlabel("Temperature [°C]", fontsize=18)
        plt.ylabel("log(P) [mmHg]", fontsize=18)
        plt.title(f"Pressure for {smiles}", fontsize=18)
        plt.legend()
        plt.tight_layout()
        # Create folder evaluation on molecules if it doesn't exist.
        if not os.path.exists(f"Data/Final model/Evaluation on molecules"):
            os.mkdir(f"Data/Final model/Evaluation on molecules")
        plt.savefig(f"Data/Final model/Evaluation on molecules/Pressure for {smiles}.png", dpi=300)
        plt.close()

'''