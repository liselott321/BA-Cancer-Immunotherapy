# BA-Cancer-Immunotherapy

## Authors
- [@liselott321](https://github.com/liselott321)
- [@tomickristina](https://github.com/tomickristina)
- [@oscario-20](https://github.com/oscario-20)

## About this Project
- satz fehlt jetzt noch
- 
### Data Sources
The primary data sources include:
- [VDJdb](https://vdjdb.cdr3.net/)
- [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/)
- [IEDB](https://www.iedb.org/)
- [10X Genomics](https://www.10xgenomics.com/datasets?query=%22A%20new%20way%20of%20exploring%20immunity%E2%80%93linking%20highly%20multiplexed%20antigen%20recognition%20to%20immune%20repertoire%20and%20phenotype%22&page=1&configure%5BhitsPerPage%5D=50&configure%5BmaxValu)

We harmonize TCR and pMHC sequences (positive binders) and generate synthetic negative examples. On the **ba** branch, we apply balanced negative sampling; on the **10X** branch, we use the raw 10X dataset.

### Data Processing @ Arina stimmt das noch?
All raw files are standardized, harmonized, and split into train/validation/test sets. Negative samples are generated synthetically (see “Data Pipeline” section).  
For details, see the [Data Pipeline 10x-allrows50-datacheck](#BA_ZHAW/data_pipeline_10x-allrows50-datacheck.ipynb) section.

### Model Architectures
We explore multiple deep-learning architectures, including:
v1-v6

### Repository Structure
`data/`: This will be used to store data locally\
`data_scripts/`: Contains all scripts related to data acquisition, preprocessing and analyzing\
`models/`: Includes different model architectures and training scripts\

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
  git clone https://github.com/tomickristina/PA-Cancer-Immunotherapy-Transformer/BA_ZHAW
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

### Train a Model
- There are four scripts to do training. Each can be run with gene or allele precision (make sure datapipeline has been run with the corresponding precision).
  - `./models/beta_physico/train_beta_physico.py`
  - `./models/beta_vanilla/train_beta_vanilla.py`
  - `./models/physico/train_physico.py`
  - `./models/vanilla/train_vanilla.py`
  - `./models/random_forest_classifiers/train_valid_test_random_forest_classifier_with_physico.py`
  - `./models/paired_vanilla_physico/train_vanilla`

- Open the train skript of your choice and head to the top of the main function.
  - set value for the variable `precision`
  - If you had to change to an absolute path in the data pipeline:
    - change `embed_base_dir` to an absolute path
    - change `physico_base_dir` to an absolute path if you train either `train_beta_physico.py` or `train_physico.py`
  - If you want to do hyperparameter tuning with Weights & Biases sweeps
    - change `hyperparameter_tuning_with_WnB` to True
  - Otherwise set the specific hyperparameter values in the train script:
  
    ```
    # ! here random hyperparameter values set !
    hyperparameters["optimizer"] = "sgd"
    hyperparameters["learning_rate"] = 5e-3
    hyperparameters["weight_decay"] = 0.075
    hyperparameters["dropout_attention"] = 0.1
    hyperparameters["dropout_linear"] = 0.45
    ```
    
  - After training one can see the checkpoint file (`.ckpt`) in the directory `checkpoints` in a directory named like the Weights & Biases run. The checkoint is saved at the point where the AP_Val metric was at its highest. Furthermore, the file with the `.pth` extension is the final model. These files are in the same directory as the training script.

