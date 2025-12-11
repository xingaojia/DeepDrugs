# DeepDrugs
DeepDrugs is a mechanism-aware deep learning framework that employs a tri-linear attention network to directly characterize how two drugs jointly act within a specific cellular context to produce synergy.
## 1 Description
Accurate prediction of drug synergy is critical for the rational design of effective combination therapies against complex diseases such as cancer. However, existing computational approaches usually characterize the effect of an individual drug on a cell line separately and then merge the effect representations of two drugs for synergy prediction, which seriously limits their abilities to capture how two drugs act together within a specific cellular environment. We introduce DeepDrugs, a mechanism-aware deep learning framework that employs a tri-linear attention network to directly characterize how two drugs jointly act within a specific cellular context to produce synergy. Extensive experiments demonstrate that DeepDrugs outperforms state-of-the-art approaches in predictive accuracy, robustness, and generalization.
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
None: For the DrugComb dataset, the early stopping patience is set to 25； For the O’Neil dataset, the early stopping patience is set to 50.
