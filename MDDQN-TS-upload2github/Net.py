#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 20:52
# @Author  : Liu Hao
# @File    : Net.py
# @Software : PyCharm
# @Information: Network
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Activation, AvgPool2D, Dropout


class Network(Model):
    def __init__(self, num_tasks, features_dim, num_filters, kernel_size, num_indicators):
        super(Network, self).__init__()
        self.features_dim = features_dim
        self.num_indicators = num_indicators
        self.num_filters = num_filters
        self.Conv2D = Conv2D(input_shape=(num_tasks, features_dim, 1), filters=num_filters,
                             kernel_size=kernel_size)
        self.BN = BatchNormalization()
        self.Act = Activation('relu')
        self.Pool = AvgPool2D(pool_size=(num_tasks - kernel_size[0] + 1, 1))
        self.Dropout1 = Dropout(0.2)
        self.Dropout2 = Dropout(0.2)
        self.Flatten = Flatten()
        # self.layer1 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())
        # self.layer2 = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1())
        self.layer1 = Dense(128, activation='relu')
        self.layer2 = Dense(128, activation='relu')
        # self.layer22 = Dense(64, activation='relu')
        self.layer3 = Dense(1)

    def call(self, inputs):
        """
        :param inputs: [[array(), [indicator_1, ...,indicator_n]], [], ...,[]]
        inputs: array(), shape(num_tasks, features_dim+1, channels=1)
        :param training:
        :param mask:
        :return:
        """
        inputs_matrix = []
        inputs_indicators = []
        num_inputs = len(inputs)
        for b in range(num_inputs):
            inputs_matrix.append(inputs[b][:, :-1, :])
            inputs_indicators.append(inputs[b][:, -1, :])

        # 构造输入卷积层的状态矩阵
        if isinstance(inputs[0], tf.Tensor):
            inputs_matrix = tf.stack(inputs_matrix, axis=0)
        else:
            inputs_matrix = np.array(inputs_matrix).astype('float32')
            inputs_matrix = tf.constant(inputs_matrix)

        conv_output = self.Conv2D(inputs_matrix)
        act_output = self.Act(conv_output)
        pool_output = self.Pool(act_output)
        # 构造输入全连接层的状态矩阵
        if isinstance(inputs[0], tf.Tensor):
            new_input = []
            for b in range(num_inputs):
                concatenated_input = pool_output[b]
                for a in range(self.num_indicators):
                    temp = tf.reshape(inputs_indicators[b][a], (1, 1, 1))
                    replicated_indicator = tf.tile(temp, multiples=[1, 1, self.num_filters])
                    concatenated_input = tf.concat([concatenated_input, replicated_indicator], axis=1)
                new_input.append(concatenated_input)
            new_input = tf.convert_to_tensor(new_input, dtype=tf.float32)
        else:
            new_input = []
            for b in range(num_inputs):
                input_tensor = pool_output[b]
                for i in range(self.num_indicators):
                    indicator_tensor = tf.constant([[[inputs_indicators[b][i][0] for _ in range(self.num_filters)]]],
                                                   dtype=tf.float32)
                    input_tensor = tf.concat([input_tensor, indicator_tensor], axis=1)
                new_input.append(input_tensor)
            new_input = tf.convert_to_tensor(new_input, dtype=tf.float32)
        flatten_output = self.Flatten(new_input)

        layer1_output = self.layer1(flatten_output)
        # layer1_output = self.Dropout1(layer1_output)
        layer2_output = self.layer2(layer1_output)
        # layer2_output = self.layer22(layer2_output)
        # layer2_output = self.Dropout2(layer2_output)
        layer3_output = self.layer3(layer2_output)
        return layer3_output

    # def call(self, inputs):
    #     """
    #     :param inputs: [[array(), [indicator_1, ...,indicator_n]], [], ...,[]]
    #     :param training:
    #     :param mask:
    #     :return:
    #     """
    #
    #     inputs_matrix = []
    #     inputs_indicators = []
    #     num_inputs = len(inputs)
    #     num_indicators = len(inputs[0][1])
    #     for b in range(num_inputs):
    #         # print('inputs[b][0]', inputs[b][0])
    #         # print('inputs[b][1]', inputs[b][1])
    #         inputs_matrix.append(inputs[b][0])
    #         inputs_indicators.append(inputs[b][1])
    #     # 构造输入卷积层的状态矩阵
    #     if isinstance(inputs[0][0], tf.Tensor):
    #         inputs_matrix = tf.stack(inputs_matrix, axis=0)
    #     else:
    #         inputs_matrix = np.array(inputs_matrix).astype('float32')
    #         inputs_matrix = tf.constant(inputs_matrix)
    #
    #     conv_output = self.Conv2D(inputs_matrix)
    #     act_output = self.Act(conv_output)
    #     pool_output = self.Pool(act_output)
    #     # 构造输入全连接层的状态矩阵
    #     if isinstance(inputs[0][0], tf.Tensor):
    #         new_input = []
    #         for b in range(num_inputs):
    #             concatenated_input = pool_output[b]
    #             for a in range(num_indicators):
    #                 temp = tf.reshape(inputs_indicators[b][a], (1, 1, 1))
    #                 replicated_indicator = tf.tile(temp, multiples=[1, 1, self.num_filters])
    #                 concatenated_input = tf.concat([concatenated_input, replicated_indicator], axis=1)
    #             new_input.append(concatenated_input)
    #         new_input = tf.convert_to_tensor(new_input, dtype=tf.float32)
    #     else:
    #         new_input = []
    #         for b in range(num_inputs):
    #             input_tensor = pool_output[b]
    #             for i in range(num_indicators):
    #                 indicator_tensor = tf.constant([[[inputs_indicators[b][i] for _ in range(self.num_filters)]]],
    #                                                dtype=tf.float32)
    #                 input_tensor = tf.concat([input_tensor, indicator_tensor], axis=1)
    #             new_input.append(input_tensor)
    #         new_input = tf.convert_to_tensor(new_input, dtype=tf.float32)
    #     flatten_output = self.Flatten(new_input)
    #     layer1_output = self.layer1(flatten_output)
    #     # layer1_output = self.Dropout(layer1_output)
    #     layer2_output = self.layer2(layer1_output)
    #     # layer2_output = self.Dropout(layer2_output)
    #     layer3_output = self.layer3(layer2_output)
    #     return layer3_output


if __name__ == '__main__':
    num_tasks = 12
    feature_dim = 6
    num_filters = 5
    kernel_size = (4, 1)
    eval1 = Network(num_tasks, feature_dim, num_filters, kernel_size)
    input_matrix1 = np.random.rand(num_tasks, feature_dim, 1)
    input_matrix2 = np.random.rand(num_tasks, feature_dim, 1)
    input_indicators1 = [99, 88]
    input_indicators2 = [11, 22]
    input1 = [input_matrix1, input_indicators1]
    input2 = [input_matrix2, input_indicators2]
    inputs = [input1, input2]
    # print('inputs', inputs)
    x = eval1.call(inputs)
    print(x)
