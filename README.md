# Pocket Captioning

Pretraining file is in the pretrain.py. It uses only MLE to train the model. 10 epochs is quite enough for covergence.

I cannot upload the pocket data due to the size. If you wish to test the code, just generate pocket toy data in the size of [bs, 14, 32, 32, 32]. Or you can download from D3D repo.

SMILES data is the labels_smiles file.

The essential libs are rdkit and pytorch.
