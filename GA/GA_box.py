#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 14:57
# @Author  : name
# @File    : GA_box.py
# @Software : PyCharm
# @Information:
import operator
import copy
import numpy as np
import pandas as pd
import random
from ladder_carbon_price import *


# CPM推算时间参数
def get_cp2(tasks_num_, successors_, predecessors_, task_duration_):
    """
    关键路径法：获取时间参数及关键路径
    :param tasks_num_: 活动数量（包括虚活动）(int)
    :param successors_: 紧后关系（Dict）
    :param task_duration_: 活动工期（dictionary）索引从1开始
    :return: es_, ef_, ls_, lf_（均为array）
    """
    # ---------------------定义变量------------------------
    es_ = np.zeros(tasks_num_, dtype='float')  # 最早开始
    ef_ = np.zeros(tasks_num_, dtype='float')  # 最早结束
    ls_ = np.zeros(tasks_num_, dtype='float')  # 最迟开始
    lf_ = np.zeros(tasks_num_, dtype='float')  # 最迟结束
    critical_path_ = []  # 关键路线

    # ----------------正向推导计算es,ef--------------------
    completed_tasks = []  # 已经结束的活动集合
    eligible_tasks = []  # 合格活动集合

    task = 1
    es_[task - 1] = 0
    ef_[task - 1] = 0
    completed_tasks.append(task)
    task_succ = successors_[task]
    for i in task_succ:
        i_pred = predecessors_[i]
        if set(completed_tasks) >= set(i_pred):  # 如果活动i的紧前活动都已经完成，则该活动加入合格活动集合
            eligible_tasks.append(i)

    while len(eligible_tasks) != 0:
        task = eligible_tasks[0]
        task_pred = predecessors_[task]
        max_ef_pred = 0
        for i in task_pred:  # 紧前活动中最迟结束时间
            if ef_[i-1] > max_ef_pred:
                max_ef_pred = ef_[i-1]
        es_[task-1] = max_ef_pred
        ef_[task-1] = es_[task-1] + task_duration_[task]

        eligible_tasks.remove(task)  # 将该活动从合格活动集合中删除
        completed_tasks.append(task)  # 将该活动添加到已完成活动集合
        # 更新合格活动集合
        task_succ = successors_[task]
        for i in task_succ:
            i_pred = predecessors_[i]
            if set(completed_tasks) >= set(i_pred):  # 如果活动i的紧前活动都已经完成，则该活动加入合格活动集合
                eligible_tasks.append(i)

    # -----------------逆向推导LS,LF-----------------------
    T = sum(task_duration_.values())
    # print('task_duration=', task_duration_)
    # print('T=', T)
    completed_tasks = []  # 已经结束的活动集合
    eligible_tasks = []  # 合格活动集合

    task = tasks_num_
    ls_[tasks_num_ - 1] = T
    lf_[tasks_num_ - 1] = T
    completed_tasks.append(task)
    task_pred = predecessors_[task]
    for i in task_pred:
        i_succ = successors_[i]
        if set(completed_tasks) >= set(i_succ):  # 如果活动i的紧前活动都已经完成，则该活动加入合格活动集合
            eligible_tasks.append(i)

    while len(eligible_tasks) != 0:
        task = eligible_tasks[0]
        task_succ = successors_[task]
        min_ls_succ = ls_[-1]
        for i in task_succ:  # 紧后活动中最早的最晚开始时间
            if ls_[i-1] < min_ls_succ:
                min_ls_succ = ls_[i-1]
        lf_[task-1] = min_ls_succ
        ls_[task-1] = lf_[task-1] - task_duration_[task]

        eligible_tasks.remove(task)  # 将该活动从合格活动集合中删除
        completed_tasks.append(task)  # 将该活动添加到已完成活动集合
        # 更新合格活动集合
        task_pred = predecessors_[task]
        for i in task_pred:
            i_succ = successors_[i]
            if set(completed_tasks) >= set(i_succ):  # 如果活动i的紧前活动都已经完成，则该活动加入合格活动集合
                eligible_tasks.append(i)

    return es_, ef_, ls_, lf_


# 检查是否符合质量要求，不符合则修复不可行解
def check_and_fix(mode_dict, Quality, wight_quality, Q, task_num):
    # 根据活动的模式获得对应质量
    quality_dict = {}
    for j in range(1, task_num+1):
        quality_dict[j] = Quality[j][mode_dict[j]]
    # 检查是否符合质量要求
    if sum([quality_dict[key] * wight_quality[key] for key in quality_dict.keys() & wight_quality.keys()]) < Q:
        # ---修复不可行解---
        meet_q_requ = False
        while not meet_q_requ:
            rand_j = random.randint(1, task_num)  # 随机选择一个活动
            if mode_dict[rand_j] == 0:
                meet_q_requ = False
            else:
                mode_dict[rand_j] -= 1  # 更换质量更高的模式
                quality_dict[rand_j] = Quality[rand_j][mode_dict[rand_j]]
            # ---检查在新的模式列表下是否达到最低质量要求---
            if sum([quality_dict[key] * wight_quality[key] for key in quality_dict.keys() & wight_quality.keys()]) >= Q:
                meet_q_requ = True
            # print(sum([quality_dict[key] * wight_quality[key] for key in quality_dict.keys() & wight_quality.keys()]), Q, meet_q_requ)
    return mode_dict


# 编码
def encoding(task_num, successors_, predecessors_, Duration, Quality, Q, wight_quality):
    # 编码，没有优先规则
    # -------给每个活动随机选择模式--------
    selected_mode = {j: 0 for j in range(1, task_num + 1)}
    selected_duration = {j: 0 for j in range(1, task_num + 1)}
    num_modes = 3

    # 先随机选择活动的执行模式，但要保证满足最低质量要求
    for j in range(1, task_num + 1):
        rand = random.randint(0, num_modes - 1)
        selected_mode[j] = rand

    # 检查在此模式列表下是否达到最低质量要求,未达到则修复
    selected_mode = check_and_fix(selected_mode, Quality, wight_quality, Q, task_num)

    # 依据模式选择活动工期
    for j in range(1, task_num + 1):
        selected_duration[j] = Duration[j][selected_mode[j]]

    # ------生成活动序列------
    _, _, _, LF = get_cp2(task_num, successors_, predecessors_, selected_duration)

    AL = [1]  # 【活动列表】已选择的活动集合，初始值为虚活动1

    while len(AL) != task_num:  # 当不是所有的活动都被选择
        # ........获得AL的紧后活动集合.........
        successors_AL = []  # AL的紧后活动集合
        for sa in AL:
            temp = successors_[sa]
            [successors_AL.append(i) for i in temp if i not in successors_AL and i not in AL]

        # ........获得合格活动集合SE.........
        SE = []  # 合格活动集合
        for task in successors_AL:  # 检查AL的每个紧后活动
            predecessors_task = predecessors_[task]  # 获得该紧后活动的紧前活动集合
            count = 0
            for i in predecessors_task:  # 检查该活动的每个紧前活动时候都在AL中
                if i in AL:
                    count += 1
                else:
                    break
            if count == len(predecessors_task):  # 如果task的全部紧前活动都在AL中
                SE.append(task)
        # 从SE中随机选一个加入AL
        rand_act_from_SE = random.choice(SE)
        AL.append(rand_act_from_SE)

    return AL, selected_mode, selected_duration


# 初始化
def initialize(popsize, task_num, successors_, predecessors_, Duration, Quality, Q, Energy_Consumption, theta, wight_quality):
    # -----------生成初始种群（活动列表、模式列表、活动工期）----------------
    # 种群的数据结构：pop = [{'al': [], 'mode': {}, 'duration': [], 'makespans': -1, 'carbon_emission': carbon_emission, 'total_cost':-1, 'es': -1, 'ef': -1, 'from': 'ga'}, ..., {}]

    pop = []
    for i in range(popsize):
        AL_, selected_mode_, selected_duration_ = encoding(task_num, successors_, predecessors_, Duration, Quality, Q, wight_quality)
        # 计算项目总能耗
        energy_consumption = 0
        for j in range(1, task_num + 1):
            energy_consumption += Energy_Consumption[j][selected_mode_[j]]
        carbon_emission = energy_consumption * theta
        individual = {'al': AL_, 'mode': selected_mode_, 'duration': selected_duration_, 'makespans': -1,
                      'carbon_emission': carbon_emission, 'total_cost': -1, 'es': -1, 'ef': -1, 'from': 'ga'}
        pop.append(individual)

    return pop


# 解码
def ssgs_decoding(successors_, predecessors_, pop, r_, s_, S_, R_, task_num, num_r):  # 【新的解码方式，提高速度】
    for individual in pop:
        AL_ = individual['al']
        selected_duration_ = individual['duration']
        PS = [1]  # 已经安排的活动集合
        ST = {j: -1 for j in range(1, task_num + 1)}  # 活动开始时间
        FT = {j: -1 for j in range(1, task_num + 1)}  # 活动结束时间
        ST[1] = 0
        FT[1] = 0

        while len(PS) < task_num:
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
            for j in range(1, task_num + 1):
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
        individual['makespans'] = FT[task_num]
        individual['es'] = ST
        individual['ef'] = FT
    return pop


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


# 计算适应值/目标函数
# def total_cost_fun(task_num, num_r, r, resource_price, energy_price, Re, Pe, carbon_price, Quota, pop, D, theta, Quality, wight_quality, Q):
def total_cost_fun(task_num, num_r, r, resource_price, energy_price, Re, Pe, Quota, pop, D, theta, gap, p_inc, init_carbon_price):
    for individual in pop:
        # 资源使用成本
        cost_resource = 0
        for j in range(1, task_num + 1):
            for x in range(num_r):
                cost_resource += individual['duration'][j] * r[j][x] * resource_price[x]
        # 能耗成本
        cost_energy = individual['carbon_emission'] / theta * energy_price
        # 提前完工奖励
        tiqian = D - individual['makespans']
        if tiqian > 0:
            reward = tiqian * Re
        else:
            reward = 0
        # 延期完成惩罚
        yanqi = individual['makespans'] - D
        if yanqi > 0:
            punishment = yanqi * Pe
        else:
            punishment = 0
        # 碳交易费用
        carbon_price = ladder_carbon_price(Quota, individual['carbon_emission'], gap, p_inc, init_carbon_price)
        Z = abs(Quota - individual['carbon_emission'])
        cost_carbon_trading = Z * carbon_price
        total_cost = cost_resource + cost_energy - reward + punishment + cost_carbon_trading

        individual['total_cost'] = total_cost
        # print('cost_resource', cost_resource, 'cost_energy', cost_energy, 'reward', reward, 'punishment', punishment,
        #       'cost_carbon_trading', -cost_carbon_trading)

    return pop


def total_cost_fun1(task_num, num_r, r, resource_price, energy_price, Re, Pe, carbon_price, Quota, individual, D, theta):
    # 资源使用成本
    cost_resource = 0
    for j in range(1, task_num + 1):
        for x in range(num_r):
            cost_resource += individual['duration'][j] * r[j][x] * resource_price[x]
    # 能耗成本
    cost_energy = individual['carbon_emission'] / theta * energy_price
    # 提前完工奖励
    tiqian = D - individual['makespans']
    if tiqian > 0:
        reward = tiqian * Re
    else:
        reward = 0
    # 延期完成惩罚
    yanqi = individual['makespans'] - D
    if yanqi > 0:
        punishment = yanqi * Pe
    else:
        punishment = 0
    # 碳交易费用
    Z = Quota - individual['carbon_emission']
    cost_carbon_trading = -Z * carbon_price
    total_cost = cost_resource + cost_energy - reward + punishment + cost_carbon_trading

    individual['total_cost'] = total_cost
    # print('cost_resource', cost_resource, 'cost_energy', cost_energy, 'reward', reward, 'punishment', punishment,
    #       'cost_carbon_trading', -cost_carbon_trading)

    return cost_resource, cost_energy, reward, punishment, cost_carbon_trading


def total_cost_fun0(task_num, num_r, r, resource_price, energy_price, Re, Pe, carbon_price, Quota, pop, D, theta):
    for individual in pop:
        total_cost = individual['carbon_emission']
        individual['total_cost'] = total_cost
    return pop


# 选择（适应值轮盘赌）
def select(pop):
    # print('----------------------')
    # print('pop')
    # for i in pop:
    #     print(i)
    new_pop = []
    # 计算种群总适应值
    POP_Total_Cost = [individual['total_cost'] for individual in pop]
    max_total_cost = max(POP_Total_Cost)
    POP_fitness = [max_total_cost - POP_Total_Cost[p] + 1 for p in range(len(POP_Total_Cost))]
    # POP_fitness = [1/x for x in POP_Total_Cost]
    total_fitness = sum(POP_fitness)
    # 计算每个个体被选中的概率
    probabilities = [POP_fitness[p] / total_fitness for p in range(len(POP_fitness))]
    # print('POP_Total_Cost', list(map(lambda x: round(x, 2), POP_Total_Cost)))
    # print('max_total_cost', max_total_cost)
    # print('POP_fitness', list(map(round, POP_fitness)))
    # print('probabilities', list(map(lambda x: round(x, 4), probabilities)))
    for i in range(len(pop)):
        # -------轮盘赌选择--------
        # 生成一个0到1之间的随机数
        r = random.random()
        # 累计概率
        c = 0
        # 遍历种群中的每个个体
        for index, individual in enumerate(pop):
            # 累加当前个体的概率
            c += probabilities[index]
            # 如果累计概率大于等于随机数，返回当前个体
            if c >= r:
                new_pop.append(individual)
                break
    # print('new_pop')
    # for i in new_pop:
    #     print(i)
    return new_pop


# 交叉
def crossover(pop, cro_prob, num_task, Quality, wight_quality, Q, Duration, Energy_Consumption, theta):  # 按模式字典交叉
    # 交叉操作：选择相邻的两个个体执行该操作
    # 活动列表交叉：单点交叉
    # 模式字典交叉：单点交叉（交叉后检查两个子代的质量，符合质量约束则保留，不符合质量要求则重新执行交叉，到达一定次数后还是不符合质量约束则停止交叉）
    new_pop = []
    while len(new_pop) != len(pop):
        father = random.choice(pop)
        mother = random.choice(pop)

        if random.random() < cro_prob:
            child1 = {'al': [], 'mode': {}, 'duration': {}, 'makespans': -1, 'carbon_emission': -1, 'total_cost': -1,
                      'es': {}, 'ef': {}, 'from': 'ga'}
            child2 = {'al': [], 'mode': {}, 'duration': {}, 'makespans': -1, 'carbon_emission': -1, 'total_cost': -1,
                      'es': {}, 'ef': {}, 'from': 'ga'}

            # ------活动列表交叉------
            point1 = random.randint(1, num_task)
            father_al = father['al']
            mother_al = mother['al']
            # 前半段
            father_al_first = father_al[:point1]
            mother_al_first = mother_al[:point1]
            # 后半段保持相对位置
            child1_al = father_al_first + [x for x in mother_al if x not in father_al_first]
            child2_al = mother_al_first + [x for x in father_al if x not in mother_al_first]
            child1['al'] = child1_al
            child2['al'] = child2_al
            # ------模式字典交叉-------
            meet_q = False  # 是否符合质量约束
            father_mode_dict = father['mode']
            mother_mode_dict = mother['mode']
            count = 0
            while not meet_q:
                point2 = random.randint(1, num_task)
                father_mode_dict_first = {k: v for k, v in father_mode_dict.items() if k <= point2}
                father_mode_dict_second = {k: v for k, v in father_mode_dict.items() if k > point2}
                mother_mode_dict_first = {k: v for k, v in mother_mode_dict.items() if k <= point2}
                mother_mode_dict_second = {k: v for k, v in mother_mode_dict.items() if k > point2}
                child1_mode_dict = copy.deepcopy(father_mode_dict_first)
                child1_mode_dict.update(mother_mode_dict_second)
                child2_mode_dict = copy.deepcopy(mother_mode_dict_first)
                child2_mode_dict.update(father_mode_dict_second)
                # 检查子代是否符合质量要求
                quality_child1 = {}
                quality_child2 = {}
                for j in range(1, num_task + 1):
                    j_mode1 = child1_mode_dict[j]
                    j_mode2 = child2_mode_dict[j]
                    quality_child1[j] = Quality[j][j_mode1]
                    quality_child2[j] = Quality[j][j_mode2]
                child1_quality = sum([quality_child1[key] * wight_quality[key] for key in quality_child1.keys() & wight_quality.keys()])
                child2_quality = sum([quality_child2[key] * wight_quality[key] for key in quality_child2.keys() & wight_quality.keys()])

                if child1_quality >= Q and child2_quality >= Q:
                    meet_q = True
                    child1['mode'] = child1_mode_dict
                    child2['mode'] = child2_mode_dict
                    child1_duration_dict = {}
                    child2_duration_dict = {}
                    for j in range(1, num_task+1):
                        child1_duration_dict[j] = Duration[j][child1_mode_dict[j]]
                        child2_duration_dict[j] = Duration[j][child2_mode_dict[j]]
                    child1['duration'] = child1_duration_dict
                    child2['duration'] = child2_duration_dict
                count += 1
                if count > 10:
                    child1['mode'] = father['mode']
                    child2['mode'] = mother['mode']
                    child1['duration'] = father['duration']
                    child2['duration'] = mother['duration']
                    break
            # 计算子代的项目碳排放
            carbon_emission1 = 0
            carbon_emission2 = 0
            for j in range(1, num_task+1):
                carbon_emission1 += Energy_Consumption[j][child1['mode'][j]] * theta
                carbon_emission2 += Energy_Consumption[j][child2['mode'][j]] * theta
            child1['carbon_emission'] = carbon_emission1
            child2['carbon_emission'] = carbon_emission2
            new_pop.append(child1)
            new_pop.append(child2)
        else:
            new_pop.append(father)
            new_pop.append(mother)

    return new_pop


# 变异
def mutation(num_task, pop, muta_prob, predecessors, successors, Quality, Q, wight_quality, Duration, Energy_Consumption, theta):
    # 通过插入操作变异活动列表
    # 通过随机改变活动的执行模式来变异模式字典（如果不符合约束则重新执行该步骤）
    new_pop = []
    count = 0
    for indi in pop:

        if random.random() < muta_prob:
            muta_indi = {'al': [], 'mode': {}, 'duration': {}, 'makespans': -1, 'carbon_emission': -1, 'total_cost': -1,
                         'es': {}, 'ef': {}, 'from': 'ga'}
            count += 1

            # # ————变异活动列表——————
            al = copy.deepcopy(indi['al'])
            point1 = random.randint(1, num_task-2)
            task = al[point1]
            pre_tasks = predecessors[task]
            suc_tasks = successors[task]
            pre_indexs = []
            suc_indexs = []
            for x in pre_tasks:
                pre_indexs.append(al.index(x))
            for y in suc_tasks:
                suc_indexs.append(al.index(y))
            left = max(pre_indexs)
            right = min(suc_indexs)
            al.remove(task)
            position = random.randint(left+1, right-1)
            al.insert(position, task)
            muta_indi['al'] = al
            # ——————变异模式字典——————
            point2 = random.randint(1, num_task-2)
            task0 = al[point2]
            rest_modes = [0, 1, 2]
            mode_dict = copy.deepcopy(indi['mode'])
            duration_dict = copy.deepcopy(indi['duration'])
            mode = mode_dict[task0]
            rest_modes.remove(mode)
            mode_dict_copy = copy.deepcopy(mode_dict)
            while len(rest_modes) != 0:
                new_mode = random.choice(rest_modes)
                rest_modes.remove(new_mode)
                mode_dict_copy[task0] = new_mode
                quality_dict = {}
                for j in range(1, num_task + 1):
                    j_mode = mode_dict_copy[j]
                    quality_dict[j] = Quality[j][j_mode]
                project_quality = sum([quality_dict[key] * wight_quality[key] for key in quality_dict.keys() & wight_quality.keys()])
                if project_quality >= Q:
                    mode_dict[task0] = new_mode
                    duration_dict[task0] = Duration[task0][new_mode]
                    break
            muta_indi['mode'] = mode_dict
            muta_indi['duration'] = duration_dict
            # 计算子代的项目碳排放
            carbon_emission = 0
            for j in range(1, num_task + 1):
                carbon_emission += Energy_Consumption[j][muta_indi['mode'][j]] * theta
            muta_indi['carbon_emission'] = carbon_emission
            new_pop.append(muta_indi)
        else:
            new_pop.append(indi)
    # print(count)
    return new_pop


# 从多个字典中找出指定键值最小的
def find_min_value(dict_list, key):
    min_value = float('inf')
    best_dict = None
    for d in dict_list:
        if key in d and d[key] < min_value:
            min_value = d[key]
            best_dict = d
    return min_value, best_dict


# 计算种群平均适应值
def average_obj_value(pop):
    avr_fit = 0
    for individual in pop:
        fit = individual['total_cost']
        if fit == -1:
            print("错误！存在未计算total_cost的个体")
            break
        else:
            avr_fit += fit
    avr_fit = avr_fit / len(pop)
    return avr_fit



