import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    
    def __init__(self, n_observations, n_hiddens_per_layer, n_actions):
        super(DQN, self).__init__()
        self.buildLayers(n_observations, n_hiddens_per_layer, n_actions)
        
    def buildLayers(self, n_observations, n_hiddens_per_layer, n_actions):
        self.layers = []
        self.layers.append(nn.Linear(n_observations, n_hiddens_per_layer[0]))
        self.layers.extend([nn.Linear(n_hiddens_per_layer[i], n_hiddens_per_layer[i+1]) for i in range(len(n_hiddens_per_layer)-1)])
        self.layers.append(nn.Linear(n_hiddens_per_layer[-1], n_actions))
        self.linear_layers = nn.ParameterList(self.layers)
        print(self.layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        y = x.clone()
        
        for i in range(len(self.layers)-1):
            linear_layer = self.layers[i]
            y = F.relu(linear_layer(y))
            
        return self.layers[-1](y) # Final shouldn't use activation function I guess