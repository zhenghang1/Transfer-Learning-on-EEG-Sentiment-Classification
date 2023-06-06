import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, hidden_units, dropout):
        """
        :param hidden_units: list of number of neurons in each layer, 
        hidden_units[0] should be input_size
        """
        super().__init__()
        self.mlp = nn.Sequential()
        # for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
        #     self.mlp.append(
        #         nn.Sequential(
        #             nn.Linear(in_dim, out_dim),
        #             nn.BatchNorm1d(out_dim),
        #             nn.ReLU(),
        #             nn.Dropout(p=dropout)
        #         )
        #     )
        for i, (in_dim, out_dim) in enumerate(zip(hidden_units[:-1], hidden_units[1:])):
            self.mlp.add_module('layer{}'.format(i),nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
    
    def forward(self, X):
        """
        X: (B, dim)
        """
        return self.mlp(X)



class Classifier(nn.Module):
    def __init__(self, hidden_units, dropout, num_class):
        super().__init__()
        if isinstance(hidden_units, list):
            self.fc = nn.Sequential(
                MLP(hidden_units, dropout),
                nn.Linear(hidden_units[-1], num_class)
            )
        elif np.isscalar(hidden_units):
            self.fc = nn.Linear(hidden_units, num_class)

    def forward(self, X):
        return self.fc(X)