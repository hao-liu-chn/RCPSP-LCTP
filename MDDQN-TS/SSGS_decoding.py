#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/15 20:26
# @Author  : Liu Hao
# @File    : SSGS_decoding.py
# @Software : PyCharm
# @Information: decode the activity-mode list
import copy
# 解码
def ssgs_decoding(successors_, predecessors_, AL_, selected_duration_, r_, s_, S_, R_, act_num,
                  num_r):  # 【新的解码方式，提高寻找活动最早可开始时间的速度】

    PS = [1]  # 已经安排的活动集合
    ST = {j: -1 for j in range(1, act_num + 1)}  # 活动开始时间
    FT = {j: -1 for j in range(1, act_num + 1)}  # 活动结束时间
    ST[1] = 0
    FT[1] = 0

    while len(PS) < act_num:
        # ........获得PS的紧后活动集合.........
        successors_PS = []  # PS的紧后活动集合
        for task in PS:
            temp = successors_[task]
            for i in temp:
                if i not in successors_PS and i not in PS:
                    successors_PS.append(i)

        # ........获得合格活动集合.........
        E = []  # 合格活动集合
        for task in successors_PS:  # 检查PA的每个紧后活动
            predecessors_task = predecessors_[task]  # 获得该紧后活动的紧前活动集合
            count = 0
            for i in predecessors_task:  # 检查该活动的每个紧前活动时候都在AL中
                if i in PS:
                    count += 1
                else:
                    break
            if count == len(predecessors_task):  # 如果task的全部紧前活动都在AL中
                E.append(task)

        # ........从E中选择出优先级最高的活动..........
        priority_E = []  # E中每个活动的优先级
        for i in E:
            priority_E.append(AL_.index(i))
        priority = min(priority_E)
        j_star = E[priority_E.index(priority)]

        # ......获得j_star的最早开始时间.......
        predecessors_j_star = predecessors_[j_star]
        ft_predecessors_j_star = []
        for i in predecessors_j_star:
            ft_predecessors_j_star.append(FT[i])
        es = max(ft_predecessors_j_star)

        check_point = []  # 获得可能需要检查的资源变化点
        for j in range(1, act_num + 1):
            if FT[j] >= es:
                check_point.append(FT[j])
        check_point.sort()

        s_need_j_star = 0  # j_star需要的空间大小
        for x in range(num_r):
            s_need_j_star += s_[x] * r_[j_star][x]

        for c in range(len(check_point)):
            start_time_point = check_point[c]  # 开始时间点
            result = satisfy_constraints(R_, S_, PS, start_time_point, r_, num_r, ST, FT, s_, j_star,
                                         s_need_j_star, backward=False)  # 检查开始时间点是否符合约束
            flag = 0
            if result:  # 如果开始时间点符合约束，检查开始到结束时间段内是否符合约束
                if c == len(check_point):
                    flag = 1
                else:
                    flag = 1
                    for x in range(c + 1, len(check_point)):
                        if check_point[x] < start_time_point + selected_duration_[
                            j_star]:  # 只检查在j*结束前的资源变化点，在其结束之后的直接跳过
                            result_check_point = satisfy_constraints(R_, S_, PS, check_point[x], r_, num_r, ST, FT, s_,
                                                                     j_star, s_need_j_star, backward=False)
                            if result_check_point == False:  # 如果在[j*开始，j*结束)区间内不满足约束则退出循环，将j*的开始时间挪到后一个检查点
                                flag = 0
                                break
                        else:
                            break

            if flag == 1:
                # print('flag=', 1)
                ST[j_star] = start_time_point
                FT[j_star] = start_time_point + selected_duration_[j_star]
                break

        # .......更新PS........
        PS.append(j_star)

    return ST, FT


# 解码时安排某个活动是否满足资源约束和空间约束
def satisfy_constraints(R_, S_, PS_, time_point, r, num_r, ST, FT, s_, j_star, s_need_j_star, backward):  # 检查当前时刻点是否满足约束条件

    # --------------计算剩余资源------------------
    remain_R = copy.deepcopy(R_)
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
                    remain_R[x] -= r[task][x]
                    remain_S -= s_[x] * r[task][x]
        else:  # 如果是从后向前推算
            if st_task < time_point <= ft_task:  # (从后向前推-->前开后闭)
                for x in range(num_r):
                    remain_R[x] -= r[task][x]
                    remain_S -= s_[x] * r[task][x]

    # -----------------检查活动在该时刻能否被安排---------------
    result = False
    trigger = 1  # 记录在时刻t是否满足约束条件
    # >>==检查资源约束==<<
    for x in range(num_r):
        if r[j_star][x] > remain_R[x]:  # 不满足资源（机器）约束
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
