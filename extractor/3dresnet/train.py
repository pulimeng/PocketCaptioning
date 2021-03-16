import os
import random
import pandas as pd
import json
import numpy as np
import time

import torch

from data_utils import VoxelDataset
from torch.utils.data import DataLoader, Subset

from model import generate_model

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: ' + str(device))

with open('./params.json') as f:
    opt = json.load(f)

save_dir = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

os.mkdir(os.path.join(opt['opath'],save_dir))
output_path = os.path.join(opt['opath'],save_dir,'ckpts')
log_path = os.path.join(opt['opath'],save_dir,'logs')
config_path = os.path.join(opt['opath'],save_dir,'configs')
os.mkdir(output_path)
os.mkdir(log_path)
os.mkdir(config_path)

d = opt
print(json.dumps(d, indent=4))
with open(os.path.join(config_path, 'params.json'), 'w') as f:
    json.dump(d, f)
    
def main(opt):
    
    if opt['manual_seed'] is None:
        opt['manual_seed'] = random.randint(1, 10000)
        print('Random Seed: ', opt['manual_seed'])
        random.seed(opt['manual_seed'])
        torch.manual_seed(opt['manual_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt['manual_seed'])
    
    criterion = torch.nn.CrossEntropyLoss()
    
    dataset = VoxelDataset(label_file=opt['lpath'], root_dir=opt['path'])
    df = pd.read_csv(opt['lpath'])
    ids = df['id'].tolist()
    labels = df['class'].tolist()
    
    seeds = np.random.randint(0, 50000, size=(50,))
    for k, sd in enumerate(seeds):
    
        trainings, validations = train_test_split(ids, test_size=0.2, stratify=labels, random_state=sd)
        train_ids = [ids.index(x) for x in trainings]
        val_ids = [ids.index(x) for x in validations]
        
        train_set = Subset(dataset, train_ids)
        train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True, drop_last=True)
        val_set = Subset(dataset, val_ids)
        val_loader = DataLoader(val_set, batch_size=opt['batch_size'], shuffle=True, drop_last=True)
        
        with open(os.path.join(log_path,'validations_{}.lst'.format(sd)), 'w') as in_strm:
            for item in validations:
                in_strm.write(item+'\n')
        
        tr_losses = np.zeros((opt['num_epochs'],))
        tr_accs = np.zeros((opt['num_epochs'],))
        val_losses = np.zeros((opt['num_epochs'],))
        val_accs = np.zeros((opt['num_epochs'],))
        
        model = generate_model(opt['depth'])
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])
        
        best_val_loss = 1e6
            
        print('===================Run {} starts==================='.format(k+1))
        for epoch in range(opt['num_epochs']):
            s = time.time()
            
            model.train()
            losses = 0
            acc = 0
            
            for i, sampled_batch in enumerate(train_loader):
                data = sampled_batch['voxel']
                data = data.to(device)
                y = sampled_batch['label'].squeeze()
                y = y.to(device)
                optimizer.zero_grad()
                output = model(data)
                # print(data.y.squeeze())
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                
                y_true = y.cpu().numpy()
                y_pred = output.data.cpu().numpy().argmax(axis=1)
                acc += accuracy_score(y_true, y_pred)*100
                losses += loss.data.cpu().numpy()

            tr_losses[epoch] = losses/(i+1)
            tr_accs[epoch] = acc/(i+1)
            
            model.eval()
            v_losses = 0
            v_acc = 0
            y_preds = []
            y_trues = []
            
            for j, sampled_batch in enumerate(val_loader):
                data = sampled_batch['voxel']
                data = data.to(device)
                y = sampled_batch['label'].squeeze()
                y = y.to(device)
                with torch.no_grad():
                    output = model(data)
                    loss = criterion(output, y)
                    
                y_pred = output.data.cpu().numpy().argmax(axis=1)
                y_true = y.cpu().numpy()
                y_trues += y_true.tolist()
                y_preds += y_pred.tolist()
                v_acc += accuracy_score(y_true, y_pred)*100
                v_losses += loss.data.cpu().numpy()
                
            cnf = confusion_matrix(y_trues, y_preds)      
            val_losses[epoch] = v_losses/(j+1)
            val_accs[epoch] = v_acc/(j+1)
            
            current_val_loss = v_losses/(j+1)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_cnf = cnf
                torch.save(model.state_dict(), os.path.join(output_path, 'best_model_{}.ckpt'.format(sd)))
            
            print('Epoch: {:03d} | time: {:.4f} seconds\n'
                  'Train Loss: {:.4f} | Train accuracy {:.4f}\n'
                  'Validation Loss: {:.4f} | Validation accuracy {:.4f} | Best validation loss {:.4f}'
                  .format(epoch+1, time.time()-s, losses/(i+1),
                  acc/(i+1), v_losses/(j+1), v_acc/(j+1), best_val_loss))
            print('Validation confusion matrix:')
            print(cnf)
            
        print('===================Run {} ends==================='.format(k+1))
        np.save(os.path.join(log_path,'train_loss_{}.npy'.format(sd)), tr_losses)
        np.save(os.path.join(log_path,'train_acc_{}.npy'.format(sd)), tr_accs)
        np.save(os.path.join(log_path,'val_loss_{}.npy'.format(sd)), val_losses)
        np.save(os.path.join(log_path,'val_acc_{}.npy'.format(sd)), val_accs)
        np.save(os.path.join(log_path,'confusion_matrix_{}.npy'.format(sd)), best_cnf)
        
        del model
    
if __name__ == "__main__":
    main(opt)