import os
import os.path as osp
import json
import time
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit import rdBase


from data_utils import Vocabulary
from dataloader import MolData
from generator.generator import Generator
from extractor.extractor import Extractor
from utils import decrease_learning_rate

rdBase.DisableLog('rdApp.error')

def pretrain(opt):
    """Trains the Generator RNN"""
      
    save_dir = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.mkdir(osp.join(opt['save_dir'], save_dir))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device: ' + str(device))

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file=opt['voc_file'])

    # Create a Dataset from a SMILES file
    moldata = MolData(opt['data_folder'], opt['threshold'], voc)
    data = DataLoader(moldata, batch_size=opt['batch_size'], shuffle=True, drop_last=True)
    
    E = Extractor(opt['gnn_layers'], opt['input_dim'], opt['hidden_dim'], opt['output_dim'])
    G = Generator(voc)
    E.to(device)
    G.to(device)
    
    optimizer = torch.optim.Adam(list(E.parameters())+list(G.parameters()), lr=opt['lr'])
    best_val_loss = 1e6
    for epoch in range(opt['num_epochs']):
        print('Start epoch {}'.format(epoch+1))
        idx = 0
        outputs = pd.DataFrame(columns=['smiles', 'valid'])
        for step, batch in tqdm(enumerate(data), total=len(data)):
            # Sample from DataLoader
            batch.to(device)
            
            # Calculate loss
            optimizer.zero_grad()
            features = E(batch)
            log_p = G.likelihood(features, batch.y.long())
            loss = - log_p.mean()

            # Calculate gradients and take a step
            loss.backward()
            optimizer.step()

            # Every 10 steps we decrease learning rate and print some information
            if step % 100 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                seqs, likelihood = G.sample(features)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                        outputs.at[idx, 'valid'] = 1
                    else:
                        outputs.at[idx, 'valid'] = 0
                    outputs.at[idx, 'smiles'] = smile
                    idx += 1
                    
                if loss.data < best_val_loss:
                    best_val_loss = loss.data.detach()
                    torch.save(E.state_dict(), osp.join(opt['save_dir'], save_dir, 'best_e_model.ckpt'))
                    torch.save(G.state_dict(), osp.join(opt['save_dir'], save_dir, 'best_g_model.ckpt'))
                tqdm.write('Epoch {:3d}   step {:3d}    loss: {:5.2f}    {:>4.1f}% valid SMILES'
                           .format(epoch, step, loss.data, 100 * valid / len(seqs)))
                
        outputs.to_csv(osp.join(opt['save_dir'], save_dir, 'epoch_{}.csv'.format(epoch+1)), index=False)
        print('Done epoch {}\n'.format(epoch+1))

if __name__ == "__main__":
    with open('./params.json') as f:
        opt = json.load(f)
    pretrain(opt)
