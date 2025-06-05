# BA-Cancer-Immunotherapy
This repository implements and extends the work presented in our Bachelor thesis at Zurich University of Applied Sciences, providing all code, data pipelines, and reproducible experiments.

## Authors
- [@liselott321](https://github.com/liselott321)
- [@tomickristina](https://github.com/tomickristina)
- [@oscario-20](https://github.com/oscario-20)

## About this Project
This project investigates transformer-based models for predicting the binding specificity between T-cell receptors (TCRs) and epitopes, a crucial task for personalized cancer immunotherapy. We systematically evaluate and develop multiple model architectures, starting from a basic transformer, and incrementally introduce genetic context information, physicochemical features, and reciprocal attention mechanisms to capture the complexity of TCR-epitope interactions.

### Data Sources
We compiled a comprehensive dataset from the following primary sources:

- **[VDJdb](https://vdjdb.cdr3.net/):** Curated database of TCR sequences with known epitope specificity.
- **[McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/):** Pathology-associated TCR sequences.
- **[IEDB](https://www.iedb.org/):** Immune Epitope Database and Analysis Resource.
- **[10X Genomics](https://www.10xgenomics.com/datasets):** High-throughput immune repertoire sequencing data.


### Data Processing @ Arina stimmt das noch?
All raw files are standardized, harmonized, and split into train/validation/test sets. Negative samples are generated synthetically (see “Data Pipeline” section).  
For details, see the [Data Pipeline 10x-allrows50-datacheck](#BA_ZHAW/data_pipeline_10x-allrows50-datacheck.ipynb) section.

#### Periodic Embedding (PE) Preparation

#### Physicochemical Latent Embedding (PLE) Preparation

To generate the PLE features, please run the following scripts in order, all located in `data_scripts/`:

1. `PLE_physico_raw.py`  
   _Extracts and preprocesses raw physicochemical data._
2. `PLE_physico_schema.py`  
   _Defines and validates the PLE data schema._
3. `PLE_physico_embedding.py`  
   _Generates the final latent embeddings for use in the model._

**You must run these scripts sequentially before training any model version that uses PLE features.**

### Model Architectures

We developed and compared the following transformer-based model versions, each adding new features or architectural components:

- **V1:** Basic multi-head attention transformer using only sequence data (TCR, epitope).
- **V2:** Adds genetic context features (TRBV, TRBJ, MHC allele embeddings).
- **V3:** Incorporates physicochemical properties of TCR and epitope sequences using periodic (sin/cos) embeddings.
- **V4:** Combines sequence data, genetic context, and physicochemical features.
- **V5:** Introduces reciprocal attention mechanisms (inspired by TSpred), genetic context, and physicochemical latent embeddings.
- **V6:** Similar 

### Repository Structure
```bash
├── WnB_Experiments_Datasets/ # Downloaded datasets from Weights & Biases Artifacts
├── configs/ # All YAML config files for model runs
├── data_scripts/ # Data processing and preparation scripts
├── models/ # Model architectures (transformers etc.)
├── models_scripts/ # Main scripts for training and testing each model version
├── results/ # Output: trained models, logs, evaluation results - (empty due to storage limitation)
├── utils/ # Utility functions and helpers
├── visualizers/ # Scripts for visualization and analysis
```
*You will need a data folder to store the data; see configs for how it should be structured.*

## Prerequisites

### Hardware
- A CUDA-compatible GPU is **strongly recommended** (e.g., NVIDIA RTX series).  
- CUDA 12.1 is tested; other CUDA versions may work but check your PyTorch installation docs.  
- Example: A GTX 1650 often runs out of memory on larger TCR–Epitope batches.

### Weights & Biases (W&B)
We use [Weights & Biases](https://wandb.ai/site) to:
- Store datasets (artifacts)
- Track hyperparameter sweeps
- Log metrics, confusion matrices, and plots

You’ll need a free W&B account.

### Conda Environment

We recommend using Anaconda/Miniconda for package, dependency, and environment management. From the root of this project, create and activate the environment:

```bash
conda env create -n BA_ZHAW --file ENV.yml
conda activate BA_ZHAW
```

Install the necessary pip packages.
```bash
pip install tidytcells
pip install peptides
```
As the pytorch installation isn't cross-compatible with every device, we suggest to reinstall it properly. First uninstall it.
```bash
conda uninstall pytorch -y
```
Now pytorch can be reinstalled. Therfore check the [Pytorch Documentation](https://pytorch.org/get-started/locally/)

#### Conda Issues
Sometimes the replication of conda environments does not work as good as we may wish. In this create a new environment with python version 3.12 or higher.
The following list should cover all the needed packages without guarantee of completeness. It will certainly prevent the vast majority of ModuleNotFound errors.
First install [Pytorch Documentation](https://pytorch.org/get-started/locally/) and then:
```
conda install numpy
pip install python-dotenv
pip install nbformat
pip install tidytcells
conda install pandas
pip install peptides
conda install wandb --channel conda-forge
conda install conda-forge::pytorch-lightning
conda install matplotlib
conda install -c conda-forge scikit-learn
conda install conda-forge::transformers
```
In some cases pytorch needs to have [sentencepiece](https://pypi.org/project/sentencepiece/) installed. When you work with cuda version 12.2 and have PyTorch installation for cuda version 12.1 installed, you will need it for sure. 
```
pip install sentencepiece
```
## Run Locally
- Clone the project
```bash
  git clone https://github.com/liselott321/BA-Cancer-Immunotherapy.git
```
- Create conda environment as explained above and use it from now on
- Open the project in the IDE of your choice
- Ensure the project is set as the root directory in your IDE. Otherwise, you may encounter path errors when running commands like %run path/to/other_notebook.ipynb.

### Run Data Pipeline
- place the [plain_data](https://www.dropbox.com/scl/fo/u38u47xq4kf51zhds16mz/AImhPziSKkpz1HS7ORnuC1c?rlkey=3we4ggnd4qjntv4gu1dgibtma&e=1&st=lc52udh3&dl=0) folder in the data folder, where the README_PLAIN_DATA.md is located.
- In order to execute the data pipeline, which harmonizes and splits data, then creates embeddings and PhysicoChemical properties, do the following:
  - Open data_pipeline.ipynb in the root folder or data_pipeline_10x-allrows50-datacheck.ipynb on the 10X branch for including 10X data 
  - set the variable precision to `precision="allele"` or `precision="gene"`
  - Run the notebook with the newly created conda environment
  - The output is placed in the `./data` folder od `./data_10x` folder
  - The final split for beta paired datasets can be found under `./data/splitted_data/{precision}/ ` or `./data_10x/splitted_data/{precision}/ `
  - Run the notebook again with different precision to create all datasets

## Train a Model
There are six main model versions (v1 through v6), each with a corresponding training and testing script under `models_scripts/v1_mha/`. Before running any script, edit its YAML config (e.g. `configs/v1_basic.yaml`, `configs/v2_cf.yaml`, etc.) to set:
- Paths to `train.tsv`, `validation.tsv`, `test.tsv`
- Embedding file locations (HDF5)
- Hyperparameters (`embed_dim`, `num_heads`, `learning_rate`, etc.)

### Available Training Scripts
- `models_scripts/v1_mha/train_v1_basic.py`
- `models_scripts/v1_mha/train_v2_cf.py`
- `models_scripts/v1_mha/train_v3_pe.py`
- `models_scripts/v1_mha/train_v4_CE_PE.py`
- `models_scripts/v1_mha/train_v5_reciprocal_attention_cf_ple.py`
- `models_scripts/v1_mha/train_v6_reciprocal_attention_cf_pe.py`

### Available Testing Scripts
- `models_scripts/v1_mha/test_v1_basic.py`
- `models_scripts/v1_mha/test_v2_cf.py`
- `models_scripts/v1_mha/test_v3_pe.py`
- `models_scripts/v1_mha/test_v4_CE_PE.py`
- `models_scripts/v1_mha/test_v5_reciprocal_attention_cf_ple.py`
- `models_scripts/v1_mha/test_v6_reciprocal_attention_cf_pe.py`

Each “vX” training script expects you to pass `--configs_path configs/vX_<name>.yaml`. The matching test script (e.g. `test_vX_<name>.py`) will load that trained model and run evaluation on the test set.

### How to Start a Training Run
1. **Change to the project root**:
```bash
cd ~/BA-Cancer-Immunotherapy
```
2. **Launch training via `nohup`** (recommended for long-running jobs). For example, to start model v1:
```bash
nohup python models_scripts/v1_mha/train_v1_basic.py   --configs_path configs/v1_basic.yaml > run_v1.log 2>&1 &
```
- All stdout/stderr is redirected into `run_v1.log`.
- To monitor progress:
```bash
tail -f run_v1.log
```
3. **Always supply your config file** with `--configs_path`, so you can tweak file locations or hyperparameters without editing the script itself.
4. When training finishes (or if you kill the process), check:
- The W&B run directory for a `.ckpt` (best validation AP) under `results/trained_models/.../epochs/`.
- A final `.pth` (or `.pt`) model file in the same directory, representing the last epoch.

### How to Run a Test Script
Once you have a saved model checkpoint (e.g. `model_epoch_*.pt` or a W&B artifact), you can run the matching test script. For example, to test v1:
```bash
python models_scripts/v1_mha/test_v1_basic.py   --configs_path configs/v1_basic.yaml   --model_path results/trained_models/v1/model_epoch_best.pt
```
- Adjust `--configs_path` and `--model_path` to point to your YAML file and the checkpoint you want to evaluate.
- The test script will load embeddings, the model weights, and produce metrics (AUC, F1, confusion matrix, etc.) that are logged to W&B and saved under `results/`.

#### Example: Running v5 Training & Testing
```bash
# Train v5
nohup python models_scripts/v1_mha/train_v5_reciprocal_attention_cf_ple.py   --configs_path configs/v5_reciprocal_attention_cf_ple.yaml > run_v5.log 2>&1 &

# Monitor training
tail -f run_v5.log

# After training completes, suppose best checkpoint is:
# results/trained_models/v5_reciprocal_attention_cf_ple/epochs/model_epoch_12.pt

# Run the v5 test
python models_scripts/v1_mha/test_v5_reciprocal_attention_cf_ple.py   --configs_path configs/v5_reciprocal_attention_cf_ple.yaml   --model_path results/trained_models/v5_reciprocal_attention_cf_ple/epochs/model_epoch_12.pt
```

#### Hyperparameter Sweeps with W&B
If you want to run a hyperparameter sweep for any version (v1–v6), make sure in your YAML:
```yaml
hyperparameter_tuning_with_WnB: True
```
Then start the same script via `nohup`. W&B will spin up multiple runs with different hyperparameter combinations defined in your sweep config (e.g. `configs/v3_pe_sweep.yaml`).

## Additional Tips
- **Monitor GPU usage**:  
```bash
nvidia-smi
```
to watch VRAM consumption and GPU load.
- **Check results directory**:  
  - `results/trained_models/` contains all epoch‐by‐epoch checkpoints.  
  - `results/roc_curve.png`, confusion matrices, and other figures are saved automatically and logged to W&B.
- **Early stopping** is implemented in each training script. If validation AP does not improve for `patience` consecutive epochs, training halts automatically.
