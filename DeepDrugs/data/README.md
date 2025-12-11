# Data Preparation
## 1 Dataset prepare
To train or test on a new dataset, please prepare the input `.npy` file with the following format: [Drug A standard SMILES, Drug B standard SMILES, Cell line DepMap ID, Synergy score].  
## 2 Drug feature

- Drug features should be extracted as described in `data/0_drug_data/graphs.py`.  
## 3 Cell line feature
- Cell line features can be obtained from the `raw/cell/` directory, including gene expression, mutation, and copy number variation data.

## 3 Usage
### 3.1 Predict with Pretrained DeepDrugs Models
Our pre-trained model for drug synergy prediction is available for download(https://zenodo.org/records/17888472). Please save it in the `save_model/`. If you want to test the pre-trained model on the DrugComb or O'Neil dataset, please modify the relevant data import sections in `utlis.py` and `models/model.py`. The specific file names and paths can be found in the `data/` directory.
Simply run:
```
 $ python main.py --mode test --saved-model ./save_model/0_fold_oneil_best_model.pth > './experiment/'$(date +'%Y%m%d_%H%M').log 2>&1
 $ python main.py --mode test --saved-model ./save_model/0_fold_Drugcomb_best_model.pth > './experiment/'$(date +'%Y%m%d_%H%M').log 2>&1
```
If you want to test your own samples, you need to preprocess the data accordingly. For details, please refer to DeepDrus-main/data/README.md.
### 3.2 Train a New Model from Scratch
To train DeepDrugs on your own dataset: First, process the data following the instructions in `DeepDrugs-main/data/README.md`. Once the data is prepared, you can proceed to run the code.
```
 $ python main.py --mode train > './experiment/'$(date +'%Y%m%d_%H%M').log 2>&1
```
