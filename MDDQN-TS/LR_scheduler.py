#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/23 11:32
# @Author  : Liu Hao
# @File    : LR_scheduler.py
# @Software : PyCharm
# @Information: 学习率余弦退火

from tensorflow.keras import optimizers
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
    def __init__(self, EPISODES, lr_max, lr_min, warmth_rate):
        super(CosineAnnealingLRScheduler, self).__init__()
        self.EPISODES = EPISODES

        if warmth_rate == 0:
            self.warm_episode = 1
        else:
            self.warm_episode = int(self.EPISODES * warmth_rate)

        self.lr_max = lr_max
        self.lr_min = lr_min

    # @tf.function
    def __call__(self, episode):
        if episode < self.warm_episode:
            lr = self.lr_max / self.warm_episode * episode
        else:
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos((episode - self.warm_episode) / self.EPISODES * np.pi))

        return lr


# 多个周期，但周期长度相同
class CosineAnnealingLRScheduler2(optimizers.schedules.LearningRateSchedule):
    def __init__(self, Ti, lr_max, lr_min):
        super(CosineAnnealingLRScheduler2, self).__init__()
        self.Ti = Ti
        self.lr_max = lr_max
        self.lr_min = lr_min

    # @tf.function
    def __call__(self, episode):
        T_cur = episode % self.Ti
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(T_cur / self.Ti * np.pi))
        return lr


# 多个周期，但周期长度递增
class CosineAnnealingLRScheduler3(optimizers.schedules.LearningRateSchedule):
    def __init__(self, Ti, lr_max, lr_min):
        super(CosineAnnealingLRScheduler3, self).__init__()
        self.Ti = Ti
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.restart = 0
        self.T_cur = 0
        self.T_multi = 2

    # @tf.function
    def __call__(self, episode):
        self.T_cur += 1
        if self.T_cur > self.Ti:
            self.Ti = self.Ti * self.T_multi
            self.T_cur = 0
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(self.T_cur / self.Ti * np.pi))
        return lr


if __name__ == '__main__':
    test_Calr = CosineAnnealingLRScheduler3(500, 1e-4, 1e-5)
    LR = []
    for episode in range(1, 7501):
        lr = test_Calr(episode)
        LR.append(lr)

    # 设置全局字体和字号
    X = list(range(1, 7501))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(4, 2.5))
    plt.plot(X, LR, c='black')
    plt.grid()
    plt.xlabel('episode')
    plt.ylabel('learning rate')
    plt.tight_layout()
    plt.show()


