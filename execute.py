# -*- coding: utf-8 -*-

"""
MNIST (手書き文字認識) の次元圧縮の題材を利用した、
Stacked Auto-Encoder の実行サンプル
"""


import csv
import logging
import numpy as np
import sys

import settings as s

import libs.utils as u

from libs.stacked_auto_encoder import StackedAutoEncoder
from libs.training_params      import TrainingParams


# Chainer MNIST data loader
# https://github.com/pfnet/chainer/blob/master/examples/mnist/data.py
from chainer_mnist import data as mnist_data


if __name__ == '__main__' :

    ### Settings
    use_gpu     = s.USE_GPU
    log_level   = s.LOG_LEVEL
    output_csv  = s.OUTPUT_CSV_FILE_NAME
    layer_sizes = s.LAYER_SIZES

    pretraining_batch_size = s.PRETRAINING_BATCH_SIZE
    pretraining_epochs     = s.PRETRAINING_EPOCHS

    finetuning_batch_size = s.FINETUNING_BATCH_SIZE
    finetuning_epochs     = s.FINETUNING_EPOCHS


    ### Logger settings
    logger    = logging.getLogger("")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    ### Layer settings
    SA = StackedAutoEncoder(layer_sizes, logger, use_gpu,
                            activation_type = u.ACTIVATION_TYPE_SIGMOID)


    ### Load MNIST data
    mnist = mnist_data.load_mnist_data()
    x_all = mnist['data'].astype(np.float32) / 255 # 0 ～ 1
    y_all = mnist['target'].astype(np.int32)


    ### Set Train and Test data
    train_size = 60000
    test_size  = 10000
    x_train, x_test = np.split(x_all, [train_size])
    y_train, y_test = np.split(y_all, [train_size])


    ### Pretraining
    pretraining_params = TrainingParams(pretraining_batch_size, pretraining_epochs)
    SA.pretraining(x_train, pretraining_params)


    ### Finetuning
    finetuning_params = TrainingParams(finetuning_batch_size, finetuning_epochs)
    SA.finetuning(x_train, finetuning_params)

    
    ### Get result
    coords = SA.getResult(x_test)
    labels = y_test

    f      = open(output_csv, 'w')
    writer = csv.writer(f, lineterminator = "\n")
    
    for i in xrange(len(labels)) :
        writer.writerow([i, labels[i], coords[i][0], coords[i][1]])
    f.close()

    print "Complete !!!"
