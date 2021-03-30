import os
import os.path as osp

import torch
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance

import rdkit.Chem as Chem
from biopandas.mol2 import PandasMol2
from torch_geometric.data import Data, Dataset

class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                root_dir : directory with all the data
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training.
    """
    def __init__(self, root_dir, threshold, voc):
        self.root_dir = root_dir
        self.threshold = threshold
        self.voc = voc
        self.ins = [x for x in os.listdir(root_dir)]
    
    def preprocess(self, pocket_path, profile_path):
        """
        Conver pocket mol2 file to pytorch geometric grpah representation
        """
        hydrophobicity = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
                               'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
                               'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
                               'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
                               'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}

        # binding probabilities are hardcoded
        binding_probability = {'ALA': 0.701, 'ARG': 0.916, 'ASN': 0.811, 'ASP': 1.015,
                                    'CYS': 1.650, 'GLN': 0.669, 'GLU': 0.956, 'GLY': 0.788,
                                    'HIS': 2.286, 'ILE': 1.006, 'LEU': 1.045, 'LYS': 0.468,
                                    'MET': 1.894, 'PHE': 1.952, 'PRO': 0.212, 'SER': 0.883,
                                    'THR': 0.730, 'TRP': 3.084, 'TYR': 1.672, 'VAL': 0.884}
        
        total_features = ['x', 'y', 'z', 'charge', 'hydrophobicity',
                               'binding_probability', 'r', 'theta', 'phi', 'sequence_entropy']
        
        node_features, edge_index, edge_attr = read_pocket(pocket_path, profile_path,
                                                           hydrophobicity, binding_probability,
                                                           total_features, self.threshold)
        return node_features, edge_index, edge_attr
    
    def __getitem__(self, i):
        folder = self.ins[i]
        pocket_path = osp.join(self.root_dir, folder, folder + '.mol2')
        mol_path = osp.join(self.root_dir, folder, folder + '.sdf')
        profile_path = osp.join(self.root_dir, folder, folder[:-2] + '.profile')
        node_features, edge_index, edge_attr = self.preprocess(pocket_path, profile_path)
        
        mol = [x for x in Chem.SDMolSupplier(mol_path)][0]
        try:
            smiles = Chem.MolToSmiles(mol)
        except:
            print(mol)
            print(folder)
        tokenized = self.voc.tokenize(smiles)
        encoded = self.voc.encode(tokenized)
        encoded = torch.tensor(encoded).long()
        y = pad_encoded(encoded)
        data = Data(x=node_features, edge_index=edge_index, y=y, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.ins)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))
        
def read_pocket(pocket_path, profile_path, hydrophobicity, binding_probability,
                features_to_use, threshold):
    """Read the mol2 file as a dataframe."""
    atoms = PandasMol2().read_mol2(pocket_path)
    atoms = atoms.df[['atom_id', 'subst_name',
                      'atom_type', 'atom_name', 'x', 'y', 'z', 'charge']]
    atoms['residue'] = atoms['subst_name'].apply(lambda x: x[0:3])
    atoms['hydrophobicity'] = atoms['residue'].apply(
        lambda x: hydrophobicity[x])
    atoms['binding_probability'] = atoms['residue'].apply(
        lambda x: binding_probability[x])

    r, theta, phi = compute_spherical_coord(atoms[['x', 'y', 'z']].to_numpy())
    if 'r' in features_to_use:
        atoms['r'] = r
    if 'theta' in features_to_use:
        atoms['theta'] = theta
    if 'phi' in features_to_use:
        atoms['phi'] = phi

    siteresidue_list = atoms['subst_name'].tolist()

    if 'sequence_entropy' in features_to_use:
        # sequence entropy data with subst_name as keys
        seq_entropy_data = extract_seq_entropy_data(
            siteresidue_list, profile_path)
        atoms['sequence_entropy'] = atoms['subst_name'].apply(
            lambda x: seq_entropy_data[x])

    if atoms.isnull().values.any():
        print('invalid input data (containing nan):')
        print(pocket_path)

    bonds = bond_parser(pocket_path)
    node_features, edge_index, edge_attr = form_graph(
        atoms, bonds, features_to_use, threshold)

    return node_features, edge_index, edge_attr

def bond_parser(pocket_path):
    f = open(pocket_path, 'r')
    f_text = f.read()
    f.close()
    bond_start = f_text.find('@<TRIPOS>BOND')
    bond_end = -1
    df_bonds = f_text[bond_start:bond_end].replace('@<TRIPOS>BOND\n', '')
    df_bonds = df_bonds.replace('am', '1')  # amide
    df_bonds = df_bonds.replace('ar', '1.5')  # aromatic
    df_bonds = df_bonds.replace('du', '1')  # dummy
    df_bonds = df_bonds.replace('un', '1')  # unknown
    df_bonds = df_bonds.replace('nc', '0')  # not connected
    df_bonds = df_bonds.replace('\n', ' ')

    # convert the the elements to integer
    df_bonds = np.array([np.float(x) for x in df_bonds.split()]).reshape(
        (-1, 4))

    df_bonds = pd.DataFrame(
        df_bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])

    df_bonds.set_index(['bond_id'], inplace=True)

    return df_bonds

def compute_edge_attr(edge_index, bonds):
    """Compute the edge attributes according to the chemical bonds."""
    sources = edge_index[0, :]
    targets = edge_index[1, :]
    edge_attr = np.zeros((edge_index.shape[1], 1))
    for index, row in bonds.iterrows():
        # find source == row[1], target == row[0]
        # minus one because in new setting atom id starts with 0
        source_locations = set(list(np.where(sources == (row[1]-1))[0]))
        target_locations = set(list(np.where(targets == (row[0]-1))[0]))
        edge_location = list(
            source_locations.intersection(target_locations))[0]
        edge_attr[edge_location] = row[2]

        # find source == row[0], target == row[1]
        source_locations = set(list(np.where(sources == (row[0]-1))[0]))
        target_locations = set(list(np.where(targets == (row[1]-1))[0]))
        edge_location = list(
            source_locations.intersection(target_locations))[0]
        edge_attr[edge_location] = row[2]
    return edge_attr


def compute_spherical_coord(data):
    """Shift the geometric center of the pocket to origin,
    then compute its spherical coordinates."""
    # center the data around origin
    center = np.mean(data, axis=0)
    shifted_data = data - center

    r, theta, phi = cartesian_to_spherical(shifted_data)
    return r, theta, phi


def cartesian_to_spherical(data):
    """
    Convert cartesian coordinates to spherical coordinates.
    Arguments:
    data - numpy array with shape (n, 3) which is the
    cartesian coordinates (x, y, z) of n points.
    Returns:
    numpy array with shape (n, 3) which is the spherical
    coordinates (r, theta, phi) of n points.
    """
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # distances to origin
    r = np.sqrt(x**2 + y**2 + z**2)

    # angle between x-y plane and z
    theta = np.arccos(z/r)/np.pi

    # angle on x-y plane
    phi = np.arctan2(y, x)/np.pi

    # spherical_coord = np.vstack([r, theta, phi])
    # spherical_coord = np.transpose(spherical_coord)
    return r, theta, phi


def extract_seq_entropy_data(siteresidue_list, profile):
    """extracts sequence entropy data from .profile"""
    # Opening and formatting lists of the probabilities and residues
    with open(profile) as profile:
        ressingle_list = []
        probdata_list = []

        # extracting relevant information
        for line in profile:
            line_list = line.split()
            residue_type = line_list[0]
            prob_data = line_list[1:]
            prob_data = list(map(float, prob_data))
            ressingle_list.append(residue_type)
            probdata_list.append(prob_data)
    ressingle_list = ressingle_list[1:]
    probdata_list = probdata_list[1:]

    # Changing single letter amino acid to triple letter
    # with its corresponding number
    count = 0
    restriple_list = []
    for res in ressingle_list:
        newres = res.replace(res, amino_single_to_triple(res))
        count += 1
        restriple_list.append(newres + str(count))

    # Calculating information entropy
    # suppress warning
    with np.errstate(divide='ignore'):
        prob_array = np.asarray(probdata_list)
        log_array = np.log2(prob_array)

        # change all infinite values to 0
        log_array[~np.isfinite(log_array)] = 0
        entropy_array = log_array * prob_array
        entropydata_array = np.sum(a=entropy_array, axis=1) * -1
        entropydata_list = entropydata_array.tolist()

    # Matching amino acids from .mol2 and .profile files and creating dictionary
    fullprotein_data = dict(zip(restriple_list, entropydata_list))
    seq_entropy_data = {k: float(
        fullprotein_data[k]) for k in siteresidue_list if k in fullprotein_data}
    return seq_entropy_data


def amino_single_to_triple(single):
    """Converts the single letter amino acid abbreviation to 
    the triple letter abbreviation."""

    single_to_triple_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                             'G': 'GLY', 'Q': 'GLN', 'E': 'GLU', 'H': 'HIS', 'I': 'ILE',
                             'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                             'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

    for i in single_to_triple_dict.keys():
        if i == single:
            triple = single_to_triple_dict[i]

    return triple

def form_graph(atoms, bonds, features_to_use, threshold):
    A = atoms.loc[:, 'x':'z']
    A_dist = distance.cdist(A, A, 'euclidean')  # the distance matrix

    # set the element whose value is larger than threshold to 0
    threshold_condition = A_dist > threshold

    # set the element whose value is larger than threshold to 0
    A_dist[threshold_condition] = 0

    result = np.where(A_dist > 0)
    result = np.vstack((result[0], result[1]))
    edge_attr = compute_edge_attr(result, bonds)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    edge_index = torch.tensor(result, dtype=torch.long)

    # normalize large features
    atoms['x'] = atoms['x']/300
    atoms['y'] = atoms['y']/300
    atoms['z'] = atoms['z']/300

    node_features = torch.tensor(
        atoms[features_to_use].to_numpy(), dtype=torch.float32)
    return node_features, edge_index, edge_attr

def pad_encoded(encoded, max_len=220):
    padded = torch.zeros((1, max_len))
    padded[0, :encoded.shape[0]] = encoded
    return padded