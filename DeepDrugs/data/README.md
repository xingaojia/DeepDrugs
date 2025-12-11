# Data Preparation
## 1 Dataset prepare
To train or test on a new dataset, please prepare the input `.npy` file with the following format: [Drug A canonical SMILES, Drug B canonical SMILES, Cell line DepMap ID, Synergy score].  
## 2 Drug feature
- Construct a `drugsmiles.npy` file where each key is a SMILES string and the corresponding value is the same SMILES string.
- Drug features should be extracted as described in `data/0_drug_data/graphs.py`.
```
 $ python graphs.py
```
## 3 Cell line feature
- Cell line features can be obtained from the `raw/cell/` directory, including gene expression, mutation, and copy number variation data.
