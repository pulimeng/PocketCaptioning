import os
import os.path as osp
import pandas as pd

train_df = pd.read_csv('../../../data/atp-heme/labels_smiles')
train_smiles = train_df['smiles'].tolist()
root = './results'
folder = 'run_2021-03-15-16_54_20'
df = pd.read_csv(osp.join(root, folder, 'epoch_13.csv'))

atp_valid = df[(df['class']==0) & (df['valid']==1)].copy()['smiles'].tolist()
heme_valid = df[(df['class']==1) & (df['valid']==1)].copy()['smiles'].tolist()
atp = list(set(atp_valid))
heme = list(set(heme_valid))

atp_new = []
for item in atp:
    if not item in train_smiles:
        atp_new.append(item)
heme_new = []
for item in heme:
    if not item in train_smiles:
        heme_new.append(item)