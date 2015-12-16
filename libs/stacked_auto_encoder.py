# -*- coding: utf-8 -*-

"""
Stacked AutoEncoder 実行クラス
"""

import logging
import math
import numpy as np

from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F

import utils as u


class StackedAutoEncoder(object) :

    def __init__(self, layer_sizes, logger = None, use_gpu = False,
                 activation_type = u.ACTIVATION_TYPE_RELU) :
        self.layer_sizes = layer_sizes
        self.use_gpu     = use_gpu

        # Activation
        self.activation = None
        if activation_type == u.ACTIVATION_TYPE_RELU :
            self.activation = F.relu
        elif activation_type == u.ACTIVATION_TYPE_SIGMOID :
            self.activation = F.sigmoid

        # Logger
        if logger is None :
            logger = logging.getLogger("")
        self.logger = logger

        # layers
        self.f_layers = self._createLayers(layer_sizes)
        self.b_layers = self._createLayers(layer_sizes, backward = True)

        # Optimizer
        # self.optimizer = optimizers.SGD()
        self.optimizer = optimizers.Adam()


    def pretraining(self, data, training_params) :
        """
        入力データを使用して Pretraining を実施
        """
        self.logger.info("Start pretraining...")

        threshold = training_params.loss_threshold

        for l_idx in xrange(len(self.f_layers)) :
            self.logger.info("Pretrining {0} and {1} layers.".format(l_idx + 1, l_idx + 2))

            ### データ設定
            train_data = None
            if l_idx == 0 :
                train_data = data
            else :
                train_data = pre_train_data


            ### Model 作成
            model = FunctionSet(
                l_in  = self.f_layers[l_idx],
                l_out = self.b_layers[l_idx],
            )
            if self.use_gpu :
                model = model.to_gpu()
            self.optimizer.setup(model)

            
            ### forward 処理設定
            def forwardWithLoss(x_data) :
                x = Variable(x_data)
                t = Variable(x_data)

                h = self.activation(model.l_in(x))
                y = self.activation(model.l_out(h))

                return F.mean_squared_error(y, t)

            def forwardWithActivation(x_data) :
                x = Variable(x_data)
                y = self.activation(model.l_in(x))
                return y.data


            ### Training
            train_size = len(train_data)
            batch_size = training_params.batch_size
            batch_max  = int(math.ceil(train_size / float(batch_size)))

            epoch = 0
            while True :
                epoch    += 1
                indexes   = np.random.permutation(train_size)
                sum_loss  = 0
                for i in xrange(batch_max) :
                    start   =  i      * batch_size
                    end     = (i + 1) * batch_size
                    x_batch = train_data[indexes[start: end]]
                    self.logger.debug("Index {0} => {1}, data count = {2}".format(start, end, len(x_batch)))

                    if self.use_gpu :
                        x_batch = cuda.to_gpu(x_batch)
                    self.optimizer.zero_grads()
                    loss = forwardWithLoss(x_batch)
                    sum_loss += loss.data * batch_size
                    loss.backward()
                    self.optimizer.update()

                epoch_loss = sum_loss / train_size
                self.logger.info("Epoch {0}, Training loss = {1}".format(epoch, epoch_loss))

                if threshold is not None and epoch_loss < threshold :
                    self.logger.info("Training loss is less then loss_threshold ({0}), stop training.".format(threshold))
                    break

                if epoch >= training_params.epochs :
                    break


            ### Set next layer data
            pre_train_data = np.zeros((train_size, self.layer_sizes[l_idx + 1]), dtype = train_data.dtype)
            for i in xrange(batch_max) :
                start   =  i      * batch_size
                end     = (i + 1) * batch_size
                x_batch = train_data[start: end]
                if self.use_gpu :
                    x_batch = cuda.to_gpu(x_batch)
                y_batch = forwardWithActivation(x_batch)
                if self.use_gpu :
                    y_batch = cuda.to_cpu(y_batch)
                pre_train_data[start: end] = y_batch
        
        self.logger.info("Complete pretraining.")


    def finetuning(self, data, training_params) :
        """
        入力データを使用して Fine tuning を実施
        """
        self.logger.info("Start finetuning...")

        threshold = training_params.loss_threshold

        ### Model 設定
        model = FunctionSet()
        for num, f_layer in enumerate(self.f_layers, 1) :
            name = "l_f{0}".format(num)
            model.__setattr__(name, f_layer)
        for num, b_layer in enumerate(self.b_layers, 1) :
            name = "l_b{0}".format(num)
            model.__setattr__(name, b_layer)

        if self.use_gpu :
            model = model.to_gpu()
        self.optimizer.setup(model)

        
        ### forward 処理設定
        def forward(x_data) :
            x = Variable(x_data)
            t = Variable(x_data)

            h = x
            for num in xrange(1, len(self.f_layers) + 1) :
                h = self.activation(model.__getitem__("l_f{0}".format(num))(h))
            for num in reversed(xrange(1, len(self.b_layers) + 1)) :
                h = self.activation(model.__getitem__("l_b{0}".format(num))(h))
            y = h

            return F.mean_squared_error(y, t)


        ### Training
        train_data = data
        train_size = len(train_data)
        batch_size = training_params.batch_size
        batch_max  = int(math.ceil(train_size / float(batch_size)))

        epoch = 0
        while True :
            epoch    += 1
            indexes   = np.random.permutation(train_size)
            sum_loss  = 0
            for i in xrange(batch_max) :
                start   =  i      * batch_size
                end     = (i + 1) * batch_size
                x_batch = train_data[indexes[start: end]]
                self.logger.debug("Index {0} => {1}, data count = {2}".format(start, end, len(x_batch)))

                if self.use_gpu :
                    x_batch = cuda.to_gpu(x_batch)

                self.optimizer.zero_grads()
                loss = forward(x_batch)
                sum_loss += loss.data * batch_size
                loss.backward()
                self.optimizer.update()

            epoch_loss = sum_loss / train_size
            self.logger.info("Epoch {0}, Training loss = {1}".format(epoch, epoch_loss))
            if threshold is not None and epoch_loss < threshold :
                self.logger.info("Training loss is less then loss_threshold ({0}), stop training.".format(threshold))
                break

            if epoch >= training_params.epochs :
                break

        self.logger.info("Complete finetuning.")


    def getResult(self, data, batch_size = 100) :
        """
        入力データをネットワークに与え、結果を取得する。
        batch_size は一度にネットワークに投げるデータ数。マシン性能により調整。
        """
        self.logger.info("Get result start.")
        
        ### Model 設定
        model = FunctionSet()
        for num, f_layer in enumerate(self.f_layers, 1) :
            name = "l_f{0}".format(num)
            model.__setattr__(name, f_layer)

        if self.use_gpu :
            model = model.to_gpu()
        self.optimizer.setup(model)

        
        ### forward 処理設定
        def forward(x_data) :
            x = Variable(x_data)
            t = Variable(x_data)

            h = x
            for num in xrange(1, len(self.f_layers)) :
                h = self.activation(model.__getitem__("l_f{0}".format(num))(h))
            y = model.__getitem__("l_f{0}".format(num + 1))(h)

            return y.data


        ### 結果取得
        test_data = data
        test_size = len(test_data)
        batch_max = int(math.ceil(test_size / float(batch_size)))

        y_data = np.zeros((test_size, self.layer_sizes[len(self.layer_sizes) - 1]), dtype = test_data.dtype)
        for i in xrange(batch_max) :
            start   =  i      * batch_size
            end     = (i + 1) * batch_size
            x_batch = test_data[start: end]
            self.logger.debug("Index {0} => {1}, data count = {2}".format(start, end, len(x_batch)))

            if self.use_gpu :
                x_batch = cuda.to_gpu(x_batch)
            y_batch = forward(x_batch)
            if self.use_gpu :
                y_batch = cuda.to_cpu(y_batch)

            y_data[start: end] = y_batch
            
        self.logger.info("Complete get result.")
        return y_data


    def _createLayers(self, sizes, backward = False) :
        """
        (private)
        連続サイズの指定により、複数の Linear layer 群を作成する。
        """
        layers     = []
        size_count = len(sizes)
        for i in xrange(size_count - 1) :
            in_size  = sizes[i]
            out_size = sizes[i + 1]
            if backward :
                in_size  = sizes[i + 1]
                out_size = sizes[i]
            layers.append(F.Linear(in_size, out_size))

        return layers


