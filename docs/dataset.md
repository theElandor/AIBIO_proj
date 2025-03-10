# Data stuff
This documentation is now necessary since the presence of different dataset classes, with different behaviours can become confusing.
## yaml configuration
Now you must specify the specific root directory of the dataset.
### Before: 
dataset_dir: /work/h2020deciderficarra_shared/rxrx1 --> deprecated, raises RuntimeError('You provided an invalid dataset path')
### Now:
dataset_dir: /work/h2020deciderficarra_shared/rxrx1/rxrx1_v1.0
### Or:
dataset_dir: /work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1

## Dataset class
The dataset class now recognizes which dataset folder you are using. This allows it to adapt the metadata to the different directory structures.

## rxrx1_v2.1
This new version of the dataset is composed of four 6x224x224 non-overlapping patches of each original 6x512x512 image, with a fifth 6x224x224 center crop image that partially overlaps
with the previous four images. A projection conv layer may be necessary to adapt the new images to standard models. Remember that xavier/kaiming initialization of these layers can
help with convergence. Do not use default initialization, if possible.

## collate functions
The collate functions now should work on arbitrarly long mean/var tuples (assuming compatible dimension between the tuples and the number of channels)
Both *channelnorm_collate* and **tuple_channelnorm_collate* were changed.

## metadata position
Since normalization metadata csvs depend on the number of channels of the images (ex: len(mean)==3 mean tuples only work on 3 channel images)
the metadata folders were now moved inside the specific dataset folder.
A copy of the original metadata folder is left outside the specific folders, to ensure a painless transition between different strategies.

/work/h2020deciderficarra_shared/rxrx1/metadata --> deprecated, will be removed
/work/h2020deciderficarra_shared/rxrx1/rxrx1_v1.0/metadata --> new position for the deprecated one
/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata --> metadata path for the new dataset folder

Please, do not mix datasets and metadatas!

### Good example:
#DIRECTORIES
checkpoint_dir: ~
dataset_dir: /work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/
load_checkpoint: ~
metadata_path: /work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/new_fancy_norm.csv


### Bad example:
#DIRECTORIES
checkpoint_dir: ~
dataset_dir: /work/h2020deciderficarra_shared/rxrx1/rxrx1_v1.0/ --> NOT COMPATIBLE WITH METADATA FILE
load_checkpoint: ~
metadata_path: /work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/new_fancy_norm.csv --> NOT COMPATIBLE WITH DATASET FOLDER

At the moment, checks were not implemented to prevent a fearless programmer from doing such an atrocity. It will raise some kind of 
error/have unexpected behaviours. Do not try.

## README.md and docs folder
It's time for documentation! Markdown links were used to provide the programmers with good...decent...acceptable documentation. You are reading it right now!

## Sleep schedule
I really need to go to sleep, it's  2AM and this python script keeps raising stupid expeptions, errors and stuff. I really hate this programming language, it's so disgusting and terrible.
I really miss C++.
I launched a job which SHOULD correctly build some kind of 6 channel normalization csv.
Goodnight.

...wait...I have to...continue documenting

## Let's assume my stupid script worked
You should see a new file: '/work/h2020deciderficarra_shared/rxrx1/rxrx1_v2.1/metadata/fullmetadata_v1.csv'
What is that sorcery?
Just give it to the dataset and use one of the two compatible collate functions.
What normalization is it?
Each experiment is grouped together and channel-wise mean/var are calculated.
The train/test/val splits were left untouched (just like Matte asked).
No stratification was perfomed (since we are using the original splits).
Since each experiment contains only one type of cell, cell-wise intra experiment normalization is implicit (this sentence sounds unnecessarily complicated).
