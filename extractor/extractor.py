import torch
import torch.nn as nn
import torch.nn.functional as F

from extractor.genet import Net

def cudait(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

class Extractor(nn.Module):
    
    def __init__(self, gnn_layers, input_dim, hidden_dim, output_dim,
                 aggregator='softmax', learn=True, msg_norm=True, mlp_layers=2,
                 jk_layer='max', process_step=2, dropout=0.0):
            
        super(Extractor, self).__init__()
        self.net = Net(gnn_layers, input_dim, hidden_dim, output_dim,
                       aggregator, learn, msg_norm, mlp_layers,
                       jk_layer, process_step, dropout)

    def forward(self, data):
        x = self.net(data)
        return x