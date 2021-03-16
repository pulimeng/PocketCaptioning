import torch
import torch.nn as nn
import torch.nn.functional as F

from extractor.voxresnet import generate_model

def cudait(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

class Extractor(nn.Module):
    
    def __init__(self, depth=10, output_size=512):
            
        super(Extractor, self).__init__()
        self.cnn = generate_model(model_depth=10, n_classes=output_size)

    def forward(self, input_voxel):
        x = self.cnn(input_voxel)
        x = F.relu(x)
        return x