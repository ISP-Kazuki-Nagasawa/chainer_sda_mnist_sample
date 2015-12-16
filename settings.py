# -*- coding: utf-8 -*-

"""
アプリケーション設定
"""

import logging


### GPU 使用フラグ
# USE_GPU = True
USE_GPU = False


### Log level
# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.INFO


### 出力ファイル名
#OUTPUT_CSV_FILE_NAME = "output.csv"
# OUTPUT_CSV_FILE_NAME = "output_500_500.csv"
OUTPUT_CSV_FILE_NAME = "output_0.csv"

### Layer size
INPUT_SIZE  = 28 * 28 # 784
OUTPUT_SIZE = 2
LAYER_SIZES = [INPUT_SIZE, 1000, 500, 250, OUTPUT_SIZE]


### Pretraining
PRETRAINING_BATCH_SIZE = 100
# PRETRAINING_EPOCHS     = 100


#PRETRAINING_EPOCHS     = 50
#PRETRAINING_EPOCHS     = 200
PRETRAINING_EPOCHS = 10

### Finetuning
FINETUNING_BATCH_SIZE = 100
# FINETUNING_EPOCHS     =  50

# FINETUNING_EPOCHS = 200
FINETUNING_EPOCHS = 10


