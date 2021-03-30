import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm
from torch_geometric.nn import GENConv, DeepGCNLayer, Set2Set, JumpingKnowledge

class Net(torch.nn.Module):
    def __init__(self, gnn_layers, input_dim, hidden_dim, output_dim,
                 aggregator, learn, msg_norm, mlp_layers,
                 jk_layer, process_step, dropout):
        super(Net, self).__init__()
        
        self.dropout = dropout
        self.gnn_layers = gnn_layers
        
        self.convs = torch.nn.ModuleList()
        
        for i in range(gnn_layers):
            if i == 0:
                conv = GENConv(in_channels=input_dim, out_channels=hidden_dim,
                               aggr=aggregator, learn_t=learn, learn_p=learn,
                               msg_norm=msg_norm, num_layers=mlp_layers)
                # norm = LayerNorm(gru_dim+1, elementwise_affine=True)
                # act = ReLU(inplace=True)
                # layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                #                  ckpt_grad=i % 3)
                self.convs.append(conv)
            else:
                conv = GENConv(in_channels=hidden_dim, out_channels=hidden_dim,
                               aggr=aggregator, learn_t=learn, learn_p=learn,
                               msg_norm=msg_norm, num_layers=mlp_layers)
                # norm = LayerNorm(hidden_dim, elementwise_affine=True)
                # act = ReLU(inplace=True)
                # layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                #                  ckpt_grad=i % 3)
                self.convs.append(conv)
                
        if jk_layer.isdigit():
            jk_layer = int(jk_layer)
            self.jump = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=jk_layer)
            self.gpl = (Set2Set(hidden_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * hidden_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)
        elif jk_layer == 'cat':
            self.jump = JumpingKnowledge(mode=jk_layer)
            self.gpl = (Set2Set(gnn_layers * hidden_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * gnn_layers * hidden_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)
        elif jk_layer == 'max':
            self.jump = JumpingKnowledge(mode=jk_layer)
            self.gpl = (Set2Set(hidden_dim, processing_steps=process_step))
            self.fc1 = Linear(2 * hidden_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.gpl.reset_parameters()
        self.jump.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = []
                
        for i in range(self.gnn_layers):
            x = self.convs[i](x, edge_index)
            xs += [x]
            
        x1 = self.jump(xs)
        x1 = self.gpl(x1, batch)
        x1 = F.relu(self.fc1(x1))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = F.relu(self.fc2(x1))
        
        return x1
