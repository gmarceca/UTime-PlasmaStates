# UTime-PlasmaStates

Implementation of the U-Time model for time-series segmentation as described 
in: https://arxiv.org/abs/1910.11162

The code was adapted to work in plasma states time series data
from https://github.com/perslev/U-Time

<img src="https://github.com/gmarceca/UTime-PlasmaStates/blob/main/UTime_detection.png" width="400" height="200" />

## Installation

<b># Installation (tested in Lac8 and spcpc395)</b>
- For CPU installation (LAC machines):
    - `source algorithms/GMUTime/UTime-PlasmaStates/install.sh cpu`
- For GPU installation (GPU required):
    - `source algorithms/GMUTime/UTime-PlasmaStates/install.sh gpu`

<b># GPUs </b>
- spcpc395 has cuda 10.0 installed and is able to use TF 2.0, add the following to your .bashrc to call cuda binaries:
    - `export PATH=/usr/local/cuda/bin:$PATH`
    - `export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH`
- TF 2.0 was compiled with cuda 10.0 and so if you want to use cuda 10.2 you would need to install TF from source
https://github.com/tensorflow/tensorflow/issues/34759
- After UTime installation, check if tensorflow-gpu is working:
    - `python -i`
    - `import tensorflow as tf`
    - `tf.test.is_gpu_available()`

## Preparation of Experiments
### Dataset
- The labeled dataset is present in the Lac8_D partition
- `/Lac8_D/DISTOOL/JET/Event_Detection/`

### Prepare dataset
`ut preprocess_plasma_state_data --data_dir new_dataset_plasma --machine TCV`

### Prepare a N-fold CV experiment
`ut cv_split_plasmastates --data_dir new_dataset_plasma --machine TCV --subject_dir_pattern cluster_* --CV 5 --selected_test_set --copy`

### Initialize a U-Time N-fold CV project
`ut init_plasmastates --name my_utime_project --model utime --data_dir new_dataset_plasma --CV 5`\
(This prepares the settings to run all folds at once. If you want to focus on a particular fold 
pass `--fold 'your_fold_number'` as an additional argument)

### Start training
`cd my_utime_project/`\
    <b># Full training (train+val / test):</b>\
    `ut train_plasma_states_detector --num_GPUs=1`\
    <b># One-fold training (train_fold / val_fold):</b>\
    `ut train_plasma_states_detector --num_GPUs=1 --fold=your_fold_number`\
    <b># Full N-fold CV training:</b>\
    `cp ../extra_scripts/run.py .`\
    python run.py

### Predict and evaluate
`ut evaluate_plasma_states --out_dir eval --data_split val_data --one_shot --overwrite --weights_file_name=@epoch_XX_val_dice_XX.h5 --fold=1`

To express the evaluation results in terms of the avg. kappa statistic (final score):\
`cp ../extra_scripts/unet_to_cnnLSTM_scores_vs_epochs.py .`\
`python unet_to_cnnLSTM_scores.py --fold=XX --epoch=XX`

- Configuration settings for TCV:
    - in_dir_eval/hparams_plasma_states_TCV.yaml:
    - in_dir_eval/dataset_configurations/dataset_1_TCV.yaml
- Configuration settings for JET:
    - in_dir_eval/hparams_plasma_states_JET.yaml:
mnfqhncoua tvnfeapjve yjxaklkuec lpfgfdmchl nxsuipjtqh wtmnxgycly rxlgneuqeg mhwuhvhvcf hdxypjxeow
wfrwhvlorm wvlktyhdye xmpaddxevw mmjhyasuxv jvninterdq ynbglqqlef ggdmarlhmb
uvarjmhtxl wsuglgsbnj ysklbbmjan ygltypdssr qwfquqwbnp bgfiwjjolo spgbmexedh dfbnimltrj lfykkewwnc
