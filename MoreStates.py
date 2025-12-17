#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/1 10:06
# @Author  : Liu Hao
# @File    : MoreStates.py
# @Software : PyCharm
# @Information: for calculating more states

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def unscheduled_NC(task_num, C, A, succ):
    """
    Calculate the complex of network with unscheduled activities
    :param task_num:
    :param C: the set of activities which are completed
    :param A: the set of activate activities
    :param succ: successors
    :return:
    """
    Scheduled_acts = C + A
    NUM_unscheduled_acts = task_num - len(Scheduled_acts)
    if NUM_unscheduled_acts != 0:
        NUM_unscheduled_arcs = 0
        for j in range(1, task_num+1):
            for x in range(len(succ[j])):
                if succ[j][x] not in Scheduled_acts:
                    NUM_unscheduled_arcs += 1
        UNC = NUM_unscheduled_arcs / NUM_unscheduled_acts
    else:
        UNC = 0
    UNC = [float(UNC)]
    return UNC


def unscheduled_RS(num_task, C, A, resource_capacity, resource_demand, single=True):
    # 未调度活动平均资源强度的【倒数】-->取值范围[0, 1] URS/UARS
    Scheduled = C+A
    Unscheduled = [j for j in range(1, num_task+1) if j not in Scheduled]
    num_res = len(resource_capacity)
    # print('Unscheduled', Unscheduled)
    URS = []
    for k in range(num_res):
        un_r = 0
        for j in Unscheduled:
            un_r += resource_demand[j][k]
        if len(Unscheduled) == 0:
            rs_k = 0
        else:
            rs_k = (un_r/len(Unscheduled))/resource_capacity[k]
        URS.append(float(rs_k))
    if single:
        URS = [sum(URS)/len(URS)]
    # print('URS', URS)
    return URS


def unscheduled_RC(num_task, C, A, resource_capacity, resource_demand, single=True):
    # 未调度活动平均资源受限程度-->取值范围[0,1] URC/UARC
    # 输出是一个list
    Scheduled = C + A
    Unscheduled = [j for j in range(1, num_task+1) if j not in Scheduled]
    num_res = len(resource_capacity)
    URC = []
    for k in range(num_res):
        un_r = 0
        un_r_sgn = 0
        for j in Unscheduled:
            if resource_demand[j][k] > 0:
                un_r_sgn += 1
                un_r += resource_demand[j][k]
        if un_r_sgn == 0:
            urc = 0  # 合适
        else:
            urc = un_r / (un_r_sgn*resource_capacity[k])
        URC.append(float(urc))
    if single:
        URC = [sum(URC)/len(URC)]
    # print('URC', URC)
    return URC


def unscheduled_AD(num_task, C, A, Duration):
    """
    [max_UAD, min_UAD]  未调度活动全部取最大/最小活动工期计算出的平均活动工期
    :param num_task:
    :param C:
    :param A:
    :param Duration: 字典，包括每个活动在不同速度模式下的活动工期
    :return: UAD = [max_UAD, min_UAD]
    """
    max_AD = sum([Duration[j][2] for j in range(1, num_task + 1)]) / num_task
    mid_AD = sum([Duration[j][1] for j in range(1, num_task + 1)]) / num_task
    min_AD = sum([Duration[j][0] for j in range(1, num_task + 1)]) / num_task

    Scheduled = C + A
    Unscheduled = [j for j in range(1, num_task + 1) if j not in Scheduled]
    if len(Unscheduled) != 0:
        max_UAD = sum([Duration[j][2] for j in Unscheduled]) / len(Unscheduled)
        min_UAD = sum([Duration[j][0] for j in Unscheduled]) / len(Unscheduled)
        max_UAD = max_UAD / max_AD
        min_UAD = min_UAD / min_AD
    else:
        max_UAD = 0
        min_UAD = 0
    UAD = [float(max_UAD), float(min_UAD)]
    # print('UAD', UAD)
    return UAD


def unscheduled_Space_strength(num_task, C, A, Acts_space, S_capacity):
    Scheduled = C + A
    Unscheduled = [j for j in range(1, num_task + 1) if j not in Scheduled]
    if len(Unscheduled) != 0:
        USS = sum([Acts_space[j] for j in Unscheduled])/len(Unscheduled)/S_capacity
    else:
        USS = 0
    USS = [float(USS)]
    # print(USS)
    return USS


if __name__ == '__main__':
    num_task = 12
    succ = {1: [2, 3, 4], 2: [7, 9, 11], 3: [7], 4: [5, 11], 5: [6, 7, 8], 6: [9], 7: [10], 8: [10], 9: [12], 10: [12],
            11: [12], 12: []}

    obs1 = ([1], [], [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    action1 = [[2, 3], [2, 0]]
    obs2 = ([1, 2], [3], [0, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    action2 = [[4], [2]]
    obs3 = ([1, 2, 3, 4], [], [0, 2, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1], [0, 0, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1])
    action3 = [[5, 11], [2, 2]]
    obs4 = ([1, 2, 3, 4, 11, 5], [], [0, 2, 0, 2, 2, -1, -1, -1, -1, -1, 2, -1], [0, 0, 0, 8, 15, -1, -1, -1, -1, -1, 15, -1])
    action4 = [[6, 8], [2, 0]]
    obs5 = ([1, 2, 3, 4, 11, 5, 6], [8], [0, 2, 0, 2, 2, 2, -1, 0, -1, -1, 2, -1], [0, 0, 0, 8, 15, 22, -1, 22, -1, -1, 15, -1])
    action5 = [[9], [0]]
    obs6 = ([1, 2, 3, 4, 11, 5, 6, 8], [9], [0, 2, 0, 2, 2, 2, -1, 0, 0, -1, 2, -1], [0, 0, 0, 8, 15, 22, -1, 22, 29, -1, 15, -1])
    action6 = [[7], [2]]
    obs7 = ([1, 2, 3, 4, 11, 5, 6, 8, 7], [9], [0, 2, 0, 2, 2, 2, 2, 0, 0, -1, 2, -1], [0, 0, 0, 8, 15, 22, 30, 22, 29, -1, 15, -1])
    action7 = [[10], [1]]
    obs8 = ([1, 2, 3, 4, 11, 5, 6, 8, 7, 9, 10], [], [0, 2, 0, 2, 2, 2, 2, 0, 0, 1, 2, -1], [0, 0, 0, 8, 15, 22, 30, 22, 29, 34, 15, -1])
    action8 = [[12], [0]]
    OBS = [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8]
    # UN_NC = []
    # for x in range(0, 8):
    #     unsch_NC = unscheduled_NC(task_num, OBS[x][0], OBS[x][1], succ)
    #     UN_NC.append(unsch_NC)
    # X = [x for x in range(0, 8)]
    # plt.plot(X, UN_NC)
    # plt.show()
    URS = unscheduled_RS(num_task, obs1[0], obs1[1], resource_capacity, resource_demand)

