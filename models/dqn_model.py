import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQNModel(nn.Module):
    
    def __init__(self, input_size: int = 16, hidden1: int = 512, 
                 hidden2: int = 512, output_size: int = 3):
        super(DuelingDQNModel, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        
        self.value_stream = nn.Linear(hidden2, 1)
        self.advantage_stream = nn.Linear(hidden2, output_size)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
