# Self-supervised Model for classification of microscopy images
To reproduce our results, use the instructions below.
## Training ViT backbone with DINO
To train the backbone we started from the original DINO repo from facebook research: https://github.com/facebookresearch/dino.
We wrote our custom data loader and integrated Cross Batch Consistency Learning from this work: https://github.com/cfredinh/CDCL.
Our setup is way simpler, since you don't need to use docker to create the training environment.
Use conda to create your virtual environment from the requirments file:
```
conda create --name dinoenv --file requirements.txt
```
Then run dino_train.sh on your slurm cluster using SBATCH.
Make sure to edit the script according your needs and work environment. Pay attention to the following parameters:
- **data_path**: path to the folder containing the rxrx1 dataset;
- **metadata_path**: path to the metadata files. You can find every metadata
    file used for this work in the **/metadata** folder. Refer to the paper
    for additional information;
- **output_dir**: path where checkpoints will be saved; 
- **load_pretrained**: path to a valid checkpoint for the selected backbone.
    Leave this field empty if you want to train a model from scratch; 
You can check the available options in main_dino.py
## Training FC Head 
The source code used to train the classification head is available in the **source** folder.
To launch the training, run with SBATCH the following script **/config/slurm/train.sh** and make sure to edit the YAML configuration file **/config/train/train_head.yaml**. Edit the following fields:
+ **backbone_weights**: path of a valid checkpoint of the selected backbone;
+ **checkpoint_dir**: path where head checkpoints will be stored;
+ **dataset_dir**: path to the dataset folder;
+ **metadata_path**: path to the metadata file;
+ **project_name** and **run_name** if you use wandb;
## UMAP Plots
To reproduce our UMAP plots, use /source/embeddings.py and specify the path of the checkpoint of interest.
## Material
Dataset: [https://www.rxrx.ai/rxrx1](https://www.rxrx.ai/rxrx1).