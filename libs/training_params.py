# -*- coding: utf-8 -*-

"""
学習に用いるパラメータの設定クラス

パラメータ
  batch_size :
    バッチ数。デフォルトは「100」。
    1回の学習実行で同時にネットワークに渡すデータ数。


  epochs :
    エポック数。デフォルトは「10」。
    loss_threshold = None の場合、指定された回数まで学習を実行する。

    loss_threshold が指定されている場合は、
    指定された回数まで学習しても loss_threshold の規定値を上回っていた場合に学習を終了する。


  loss_threshold :
    学習lossの下限値。デフォルトは「None」。
    値が指定された時、学習lossが指定値を下回ったら学習を終了する。

    学習lossがいつまでも指定値を下回らない場合、epochs の値以上学習が行われたら学習を終了する。
"""


class TrainingParams(object) :

    def __init__(self, 
                 batch_size = 100, epochs = 10, loss_threshold = None) :
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.loss_threshold = loss_threshold

