# DeepDrugs
DeepDrugs is a mechanism-aware deep learning framework that employs a tri-linear attention network to directly characterize how two drugs jointly act within a specific cellular context to produce synergy.
## 1 Description
Accurate prediction of drug synergy is critical for the rational design of effective combination therapies against cancer. However, existing computational approaches usually characterize the effect of an individual drug on a cell line separately and then merge the effect representations of two drugs for synergy prediction, which seriously limits their abilities to capture how two drugs act together within a specific cellular environment. We introduce DeepDrugs, a mechanism-aware deep learning framework that employs a tri-linear attention network to directly characterize how two drugs jointly act within a specific cellular context to produce synergy. Extensive experiments demonstrate that DeepDrugs outperforms state-of-the-art approaches in predictive accuracy, robustness, and generalization.
## 2 Installation
### 2.1 System requirements
* GPU:A100
* CUDA>=11.6

### 2.2 Software Dependencies
#### Create an environment and Install DeepDrugs dependencies
We highly recommend to use a virtual environment for the installation of DeepDrugs and its dependencies. A virtual environment can be created and deactivated as follows by using conda(https://conda.io/docs/):
```
  # create
  $ conda create -n deepdrugs python=3.9
  # activate
  $ conda activate deepdrugs
```
Install pytorch 1.13.1 (For more details, please refer to https://pytorch.org/)
```
 $ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Install PyTorch Geometric (for CUDA 11.6):
```
  $ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
  $ pip install torch-geometric==2.6.1
```
Other core dependencies:
```
 $ pip install numpy==1.22.4 pandas==2.2.1 scikit-learn==1.6.1 rdkit==2025.3.3 mmcv-full==1.5.0 dgllife==0.3.2
 $ pip install dgl-cu116 -f https://data.dgl.ai/wheels/repo.html
```
## 3 Usage
### 3.1 Predict with Pretrained DeepDrugs Models
Our pre-trained model for drug synergy prediction is available for download((https://doi.org/10.5281/zenodo.18830842)). Please save it in the `save_model/`directory.
1. For the data you want to test, obtain the canonical SMILES strings for the drugs from the ChEMBL database and standardize them using RDKit. Obtain the corresponding DepMap IDs for the cell lines from the DepMap database. Save your data in the following format as a `npy` file named `0_fold_test_{user}.npy`:
```
 [drug_A_canonical_smi, drug_B_canonical_smi, DepMap_ID, synergy_score]
```
You can create a custom folder `data/user` under the `data/` directory and place your `.npy` file there.
(Since you want to predict synergy scores, the `synergy_score` values in the file can be set to 0.)

2. We provide `data/0_drug_data/graphs.py` for generating molecular graphs, and `data/raw_data/gene1061.txt` to help select multi-omics data for the 1061 genes.
3. You also need to prepare the following files for your custom dataset:
 1) **`drug_smiles.npy`**  
   - A dictionary mapping: `[drug_canonical_smi: drug_canonical_smi]`  
   - Save this file in: `data/user/`
 2) **`cell_feature_1061genes.npy`**  
   - A dictionary mapping: `[DepMap_ID: feature_matrix]`  
   - Save this file in: `data/user/`
4. Before prediction, modify the relevant data import sections in `utils.py` and `models/model.py` to match your file paths and names.
Once the above steps are completed, run the following commands to predict:
```
 $ python main.py --mode test --saved-model ./save_model/0_fold_Drugcomb_best_model.pth > './experiment/'$(date +'%Y%m%d_%H%M').log 2>&1
```
After running the prediction, the model will generate a file named: `pred_vs_true_synergy_score.csv`.This file contains the predicted synergy scores along with the true values for your test dataset.
### 3.2 Train a New Model from Scratch
If you want to train your own DeepDrugs model, you also need to process your dataset following the same steps described above for prediction. Specifically:
1. Prepare your dataset and split it into **training, validation, and test sets**.  
   - Save each set in your own directory (e.g., `data/user/`) in the required `.npy` format.
2. If your dataset is **large**, we recommend setting **early stopping patience = 25** during training.  
   If your dataset is **small**, you can set **early stopping patience = 50** to ensure sufficient training.

Once your data is ready, run the training command:

```bash
$ python main.py --mode train > './experiment/'$(date +'%Y%m%d_%H%M').log 2>&1
