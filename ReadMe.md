# Codes for Cdiscount kaggle competition of NullPointerException team
This document is for kaggle  Cdiscount competition. All codes are written by Python3.5.

## Directory & File structure
.
├── code                            # Code files
|   ├── model                       # model package
|   |   ├── loss.py
|   |   ├── lr_schedule.py
|   |   └── xception.py
|   ├── utils                       # utils package
|   |   ├── callbacks.py
|   |   ├── iterator.py
|   |   ├── sysmonitor.py           # (optional) use for monitoring CPU & GPU status
|   |   └── utils.py                # default settings
|   ├── feature_extractor.py
|   ├── fine_tuning.py
|   ├── predict_with_snapshot.py
|   ├── prediction.py
|   ├── preprocessing.py
|   ├── split_validation.py
|   ├── train_with_branch.py
|   └── training.py
├── data                            # Data files
|   ├── input                       # Original data files
|   |   ├── category_names.csv
|   |   ├── sample_submission.csv
|   |   ├── test.bson
|   |   └── train.bson
|   ├── logs
|   |   └── ...
|   ├── results
|   |   └── ...
|   ├── utils
|   |   └── ...
|   └── weights
|       └── ...
├── source_code                     # Unused files, only for reference
|   └── ...
└── ReadMe.md

## Instruction
1. Put your input files into /data/input folder.
2. Preprocess the dataset using `preprocessing.py`
3. Split validation set from train.bson using `split_validation.py`
4. Train the model using `training.py`
5. Make prediction and submission using `prediction.py` or `prediction_with_snapshot.py`

## Team Members
LZM

ZJY

MTJ

## Requirements
1. Tensorflow 1.3.0
2. Keras 2.0.9 (2.0.9 support multi-gpu)
3. Pymongo 3.5.1 (use for import bson, don't really import pymongo)
4. Pandas
5. Numpy
6. Matplotlib
7. h5py

## Citations
All codes from internet are listed in source_code folder
* [Keras generator for reading directly from BSON](https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson)
* [Snapshot Ensembles in Keras](https://github.com/titu1994/Snapshot-Ensembles)
* [Accelerating Deep Learning with Multiprocess Image Augmentation in Keras](https://github.com/stratospark/keras-multiprocess-image-data-generator)
* [Cyclical Learning Rate (CLR)](https://github.com/bckenstler/CLR)
* [SENet-Caffe](https://github.com/shicai/SENet-Caffe)
* [SE-ResNet-50 in Keras](https://gist.github.com/hollance/8d30bf5c1622036d16c4f27bd0ec88bf)
