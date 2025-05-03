# BA-Cancer-Immunotherapy

## Authors
- [@liselott321](https://github.com/liselott321)
- [@tomickristina](https://github.com/tomickristina)
- [@oscario-20](https://github.com/oscario-20)

## About this Project
This project builds upon the foundational work of a Bachelor's thesis from the [BA_ZHAW repository](https://github.com/vegger/BA_ZHAW) and our last project [PA-Cancer-Immunotherapy-Transformer](https://github.com/tomickristina/PA-Cancer-Immunotherapy-Transformer). 


## Credits
This project builds upon the work of the [BA_ZHAW project](https://github.com/vegger/BA_ZHAW). Portions of the README, data scripts, and model architectures are adapted from that repository.

### Data Sources
The primary sources of data include [VDJdb](https://vdjdb.cdr3.net/), [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/), and [IEDB](https://www.iedb.org/), which provide sequences and true postitive binding data for TCRs and pMHCs. Additionally we added [10X](https://www.10xgenomics.com/datasets?query=%22A%20new%20way%20of%20exploring%20immunity%E2%80%93linking%20highly%20multiplexed%20antigen%20recognition%20to%20immune%20repertoire%20and%20phenotype%22&page=1&configure%5BhitsPerPage%5D=50&configure%5BmaxValu).

### Data Processing
The data is standardized, harmonized, and split into training, validation, and test sets. Negative samples are synthetically generated to ensure a balanced dataset on branch ba but on branch 10X we used the new 10X dataset. The [Data Pipeline 10x-allrows50-datacheck](#BA_ZHAW/data_pipeline_10x-allrows50-datacheck.ipynb) section explains how you can run the data pipeline locally.

### Model Architectures
Various deep learning architectures are explored, including attention-based models. The [Model Training](#train-a-model) section explains how the training works in this project. For 10X Model training go on branch 10X and open there Model Training.

### Repository Structure
`data/`: This will be used to store data locally\
`data_scripts/`: Contains all scripts related to data acquisition, preprocessing and analyzing\
`models/`: Includes different model architectures and training scripts\

## Prerequisites
The following requirements must be met in order to work with this project.

### Hardware
Make sure you have a proper GPU and CUDA (version 12.1) installed. Other CUDA versions may work too but the [PyTorch](https://pytorch.org/get-started/locally/) installation can lead to problems. The NVIDIA GeForce GTX 1650 for example is not sufficient and leads to 'CUDA out of memory' issues. 

### Weights and Biases account
The [Weights and Biases](https://wandb.ai/site) account is used as MLOps Plattform to store datasets and do some model tuning.


### Conda Environment
We recommend to have Anaconda installed which provides package, dependency, and environment management for any language. To import the conda environment, execute the following command in the root folder of this project and activate it.
The name of the environment should be preserved and is called BA_ZHAW.
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

