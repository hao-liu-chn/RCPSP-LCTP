#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/27 22:01
# @Author  : Liu Hao
# @File    : total_float.py
# @Software : PyCharm
# @Information: calculate total float, look for critic activities
import copy
import numpy as np
import pandas as pd



def satisfy_constraints(resource_capacity, S_, PS_, time_point, resource_demand, num_r, ST, FT, s_, j_star, s_need_j_star, backward):  # 检查当前时刻点是否满足约束条件
    # --------------计算剩余资源------------------
    remain_R = copy.deepcopy(resource_capacity)
    remain_S = copy.deepcopy(S_)
    for task in PS_:  # 检查已经安排的活动集合中的每个活动
        st_task = ST[task]
        ft_task = FT[task]
        if not backward:  # 如果是从前向后推算
            '''
            (为什么前闭后开？当活动2开始时间为0结束时间为3的时候，
            在第3秒末，活动2占用的资源释放出来，
            活动3可以从第3秒(第3秒末第4秒处)开始)
            '''
            if st_task <= time_point < ft_task:  # 如果在t时刻PS中的task正在执行 (从前向后推-->前闭后开)
                for x in range(num_r):
                    remain_R[x] -= resource_demand[task][x]
                    remain_S -= s_[x] * resource_demand[task][x]
        else:  # 如果是从后向前推算
            if st_task < time_point <= ft_task:  # (从后向前推-->前开后闭)
                for x in range(num_r):
                    remain_R[x] -= resource_demand[task][x]
                    remain_S -= s_[x] * resource_demand[task][x]

    # -----------------检查活动在该时刻能否被安排---------------
    result = False
    trigger = 1  # 记录在时刻t是否满足约束条件
    # >>==检查资源约束==<<
    for x in range(num_r):
        if resource_demand[j_star][x] > remain_R[x]:  # 不满足资源（机器）约束
            trigger = 0
            break

    if trigger == 1:  # 已经满足了资源约束再检查是否满足空间约束
        # >>==检查空间约束==<<
        if s_need_j_star > remain_S:  # 不满足空间约束
            result = False
        else:  # 符合约束条件
            result = True
    # print('结果',result,'time_point=', time_point, 'remain_R=', remain_R, 'remain_S=', remain_S)
    return result


# [正式]计算总时差（资源约束与工作面约束）
def calculate_total_float(ES, EF, successors, predecessors, resource_capacity, resource_demand, S, s):
    NUM_ACTS = len(ES)
    DURATIONS = {j: EF[j]-ES[j] for j in range(1, NUM_ACTS+1)}
    num_r = len(resource_capacity)
    # 虚活动从1开始编号
    NUM_ACTS = len(ES)
    TF = {j: None for j in range(1, NUM_ACTS+1)}
    T = EF[NUM_ACTS]  # 项目工期
    LF = {j: None for j in range(1, NUM_ACTS+1)}  # 最迟结束时间
    LS = {j: None for j in range(1, NUM_ACTS+1)}  # 最迟开始时间
    j_star = [NUM_ACTS]  # 要被安排的活动
    PS = [NUM_ACTS]  # 局部调度计划
    E = []  # 合格活动集合
    LF[NUM_ACTS] = T
    LS[NUM_ACTS] = LF[NUM_ACTS] - DURATIONS[NUM_ACTS]
    count = 1
    while len(PS) != NUM_ACTS:
        count += 1
        if count > NUM_ACTS + 10:  # 防止出错陷入死循环
            break
        # ---------更新E-----------
        for j in j_star:
            j_pred = predecessors[j]  # 活动j的紧前活动集合
            for p in j_pred:  # 检查j的每个紧前活动
                p_succ = successors[p]
                if set(p_succ).issubset(set(PS)):  # 如果p的全部紧后活动都在PS中
                    if p not in E:
                        E.append(p)
        # ----------选择j_star------------
        ef_E = {}
        j_star = []  # 清空上一代的j_star
        for e in E:
            ef_E[e] = EF[e]

        for e in E:
            if ef_E[e] == max(ef_E.values()):
                j_star.append(e)

        for j in j_star:  # 从E中删除j_star
            E.remove(j)

        # ----------寻找j_star的lf------------
        for j in j_star:
            # 计算活动j需要的空间大小
            s_need_j = 0
            for x in range(num_r):
                s_need_j += s[x] * resource_demand[j][x]

            # 计算可能的lf的最大值
            j_succ = successors[j]
            ls_j_succ = []
            for js in j_succ:
                ls_j_succ.append(LS[js])
            temp_lf_j = min(ls_j_succ)

            # 获得资源/可用工作面的变化点
            check_points = [temp_lf_j]  # 资源/可用工作面的变化点（在指定区间内的PS中活动的ls点）
            for ps in PS:
                # print('ps', ps, 'j', j, 'ES[j]', ES[j], 'ps', ps, 'LS[ps]', LS[ps])
                if LS[ps] not in check_points and ES[j] <= LS[ps] <= temp_lf_j:
                    check_points.append(LS[ps])
            check_points.sort(reverse=True)  # 按照从大到小排序

            # 依次检查每个资源，找到符合约束的时间点
            # print('-------')
            # print('j', j, 'PS', PS)
            # print('LS', LS)
            # print('LF', LF)
            # print('check_points', check_points)
            for time_point in check_points:

                result = satisfy_constraints(resource_capacity, S, PS, time_point, resource_demand, num_r, LS, LF, s, j, s_need_j, backward=True)
                # print('time_point', time_point, 'result', result)
                if result:
                    lf_j = time_point
                    ls_j = lf_j - DURATIONS[j]
                    LF[j] = lf_j
                    LS[j] = ls_j
                    PS.append(j)
                    break
                if time_point == check_points[-1] and result is False:
                    print('【错误！】def calculate_total_float()出错了，找不到lf')

    # 计算总时差
    for j in range(1, NUM_ACTS+1):
        TF[j] = LF[j] - EF[j]
    Critic_Acts = [j for j in range(1, NUM_ACTS+1) if TF[j]==0]
    return LS, LF, TF, Critic_Acts




if __name__ == '__main__':
    # an example for test
    ES = {1: 0, 2: 0, 3: 21, 4: 10, 5: 21, 6: 10, 7: 31, 8: 36, 9: 48, 10: 48, 11: 21, 12: 59}
    EF = {1: 0, 2: 10, 3: 28, 4: 21, 5: 31, 6: 16, 7: 36, 8: 48, 9: 54, 10: 59, 11: 27, 12: 59}
    successors = {1: [2, 3, 4], 2: [6, 7], 3: [9, 10], 4: [5, 11], 5: [7, 8], 6: [10, 11], 7: [9], 8: [10], 9: [12], 10: [12], 11: [12], 12: []}
    predecessors = {1: [], 2: [1], 3: [1], 4: [1], 5: [4], 6: [2], 7: [2, 5], 8: [5], 9: [3, 7], 10: [3, 6, 8], 11: [4, 6], 12: [9, 10, 11]}
    resource_capacity = [13, 9, 28, 34]
    resource_demand = {1: [0, 0, 0, 0], 2: [0, 8, 9, 0], 3: [0, 3, 0, 5], 4: [0, 7, 10, 0], 5: [8, 0, 0, 7], 6: [6, 0, 7, 0], 7: [10, 0, 6, 0], 8: [9, 0, 7, 0], 9: [5, 0, 0, 6], 10: [5, 0, 4, 0], 11: [2, 0, 0, 6], 12: [0, 0, 0, 0]}
    S = 124
    s = [4, 5, 3, 3]
    LS, LF, TF, Critic_Acts = calculate_total_float(ES, EF, successors, predecessors, resource_capacity, resource_demand, S, s)
    print('LS', LS)
    print('LF', LF)
    print('TF', TF)
    print('Critic Acts', Critic_Acts)
