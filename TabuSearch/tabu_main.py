#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/7 10:00
# @Author  : Liu Hao
# @File    : tabu_main.py
# @Software : PyCharm
# @Information: tuning parameters for TS

"""
++++++++++++++++++++++++++++++++++++++++++++++++++
+  This version corrects errors in the previous  +
+  version and adds aspiration criterion         +
++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import copy
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from read_carbonRCPSP_data3 import read_carbon_rcpsp_file
from tqdm import tqdm
import os
import Caculate as Ca
from ladder_carbon_price import *
from SSGS_decoding import ssgs_decoding
from total_float import calculate_total_float


def get_cp2(tasks_num_, successors_, predecessors_, task_duration_):

    es_ = np.zeros(tasks_num_, dtype='float')
    ef_ = np.zeros(tasks_num_, dtype='float')
    ls_ = np.zeros(tasks_num_, dtype='float')
    lf_ = np.zeros(tasks_num_, dtype='float')
    critical_path_ = []


    completed_tasks = []
    eligible_tasks = []

    task = 1
    es_[task - 1] = 0
    ef_[task - 1] = 0
    completed_tasks.append(task)
    task_succ = successors_[task]
    for i in task_succ:
        i_pred = predecessors_[i]
        if set(completed_tasks) >= set(i_pred):
            eligible_tasks.append(i)

    while len(eligible_tasks) != 0:
        task = eligible_tasks[0]
        task_pred = predecessors_[task]
        max_ef_pred = 0
        for i in task_pred:
            if ef_[i-1] > max_ef_pred:
                max_ef_pred = ef_[i-1]
        es_[task-1] = max_ef_pred
        ef_[task-1] = es_[task-1] + task_duration_[task]

        eligible_tasks.remove(task)
        completed_tasks.append(task)

        task_succ = successors_[task]
        for i in task_succ:
            i_pred = predecessors_[i]
            if set(completed_tasks) >= set(i_pred):
                eligible_tasks.append(i)

    T = sum(task_duration_.values())
    # print('task_duration=', task_duration_)
    # print('T=', T)
    completed_tasks = []
    eligible_tasks = []

    task = tasks_num_
    ls_[tasks_num_ - 1] = T
    lf_[tasks_num_ - 1] = T
    completed_tasks.append(task)
    task_pred = predecessors_[task]
    for i in task_pred:
        i_succ = successors_[i]
        if set(completed_tasks) >= set(i_succ):
            eligible_tasks.append(i)

    while len(eligible_tasks) != 0:
        task = eligible_tasks[0]
        task_succ = successors_[task]
        min_ls_succ = ls_[-1]
        for i in task_succ:
            if ls_[i-1] < min_ls_succ:
                min_ls_succ = ls_[i-1]
        lf_[task-1] = min_ls_succ
        ls_[task-1] = lf_[task-1] - task_duration_[task]

        eligible_tasks.remove(task)
        completed_tasks.append(task)

        task_pred = predecessors_[task]
        for i in task_pred:
            i_succ = successors_[i]
            if set(completed_tasks) >= set(i_succ):
                eligible_tasks.append(i)

    return es_, ef_, ls_, lf_


def encoding(task_num, successors_, predecessors_, Duration):
    selected_mode = {j: 0 for j in range(1, task_num + 1)}
    selected_duration = {j: 0 for j in range(1, task_num + 1)}
    num_modes = 3

    for j in range(1, task_num + 1):
        rand = random.randint(0, num_modes - 1)
        selected_mode[j] = rand

    for j in range(1, task_num + 1):
        selected_duration[j] = Duration[j][selected_mode[j]]

    _, _, _, LF = get_cp2(task_num, successors_, predecessors_, selected_duration)

    AL = [1]

    while len(AL) != task_num:
        successors_AL = []
        for sa in AL:
            temp = successors_[sa]
            [successors_AL.append(i) for i in temp if i not in successors_AL and i not in AL]


        SE = []
        for task in successors_AL:
            predecessors_task = predecessors_[task]
            count = 0
            for i in predecessors_task:
                if i in AL:
                    count += 1
                else:
                    break
            if count == len(predecessors_task):
                SE.append(task)
        rand_act_from_SE = random.choice(SE)
        AL.append(rand_act_from_SE)

    return AL, selected_mode, selected_duration


def get_total_cost(act_list, mode_list):
    duration_list = {j: Duration[j][mode_list[j]] for j in range(1, task_num + 1)}
    ST, FT = ssgs_decoding(successors, predecessors, act_list, duration_list, resource_demand, s, S, resource_capacity,
                           task_num, num_r)
    res_cost = 0
    energy_cost = 0
    all_carbon_emission = 0
    U = range(1, task_num + 1)
    for task in U:
        mod = mode_list[task]
        res_cost += sum(np.asarray(resource_price) * np.asarray(resource_demand[task])) * Duration[task][mod]
        energy_cost += energy_price * Energy_Consumption[task][mod]
        all_carbon_emission += Energy_Consumption[task][mod] * theta

    T = FT[task_num]
    if T <= D:
        ms_reward = (D - T) * Re
        ms_punish = 0
    else:
        ms_reward = 0
        ms_punish = (T - D) * Pe

    carbon_price = ladder_carbon_price(Quota, all_carbon_emission, gap, p_inc, init_carbon_price)
    carbon_trading = abs(Quota - all_carbon_emission) * carbon_price
    total_cost = res_cost + energy_cost - ms_reward + ms_punish + carbon_trading
    return total_cost


def improved_swap2(act_list):
    new_act_list = copy.deepcopy(act_list)
    same = True
    count = 0
    tabu_action = [None, None]
    while same:
        pos1 = random.randint(1, task_num - 2)
        act1 = act_list[pos1]
        Succ_act1 = successors[act1]
        Succ_index = [act_list.index(x) for x in Succ_act1]
        right_pos = min(Succ_index) - 1
        Pred_Act1 = predecessors[act1]
        Pred_index = [act_list.index(y) for y in Pred_Act1]
        left_pos = max(Pred_index) + 1
        count += 1
        if right_pos - left_pos > 2:
            pos2 = random.randint(left_pos, right_pos)
            act2 = new_act_list[pos2]
            new_act_list[pos1] = act2
            new_act_list[pos2] = act1
            if new_act_list != act_list:
                same = False
                tabu_action = [act1, act2]
        if count > 50:
            break
    return new_act_list, tabu_action


def check_tabu_list_act(act1, act2, Tabu_List_act):
    tabu = False
    if len(Tabu_List_act) != 0:
        for X in Tabu_List_act:
            if [act1, act2] == X[:-1] or [act2, act1] == X[:-1]:
                tabu = True
                break
    return tabu


def check_tabu_list_mode(act, mode, Tabu_List_mode):
    tabu = False
    if len(Tabu_List_mode) != 0:
        for X in Tabu_List_mode:
            if [act, mode] == X[:-1]:
                tabu = True
                break
    return tabu


def optimize_modes_cp(mode_list, ES, EF, Tabu_List_mode):
    LS, LF, TF, Critic_Acts = calculate_total_float(ES, EF, successors, predecessors, resource_capacity, resource_demand, S, s)
    Not_Critic_Acts = [j for j in range(1, task_num+1) if j not in Critic_Acts]
    new_mode_list = copy.deepcopy(mode_list)
    tabu_action = [None, None]
    if len(Not_Critic_Acts) != 0:
        stop = 50
        count = 1
        while count <= stop:
            count += 1
            act = random.choice(Not_Critic_Acts)
            st = ES[act]
            COST = []
            for mode in range(0, 3):
                ft = st + Duration[act][mode]
                temp_mode_list = copy.deepcopy(mode_list)
                temp_mode_list[act] = mode
                if ft <= LF[act]:
                    act_res_cost = sum(resource_demand[act][y] * resource_price[y] * Duration[act][mode] for y in range(0, 4))
                    act_energy_cost = energy_price * Energy_Consumption[act][mode]
                    all_carbon_emission = sum([Energy_Consumption[j][temp_mode_list[j]] for j in range(1, task_num+1)]) * theta
                    carbon_price = ladder_carbon_price(Quota, all_carbon_emission, gap, p_inc, init_carbon_price)
                    act_emission_cost = carbon_price * theta * Energy_Consumption[act][mode]
                    cost = act_res_cost + act_energy_cost + act_emission_cost
                else:
                    cost = 100000000000
                COST.append(cost)
            best_mode = COST.index(min(COST))
            tabu = check_tabu_list_mode(act, best_mode, Tabu_List_mode)
            if not tabu:
                new_mode_list[act] = best_mode
                tabu_action = [act, best_mode]
                break
            else:
                pass

        # if best_cost < old_cost:
        #     print('good')
        # else:
        #     print('not useful')
    return new_mode_list, tabu_action


def optimize_modes(mode_list):
    new_mode_list = copy.deepcopy(mode_list)
    U = list(range(1, task_num))
    act = random.choice(U)
    original_mode = mode_list[act]
    modes_for_choose = [i for i in range(0, 3) if i != original_mode]
    chosen_mode = random.choice(modes_for_choose)
    new_mode_list[act] = chosen_mode
    tabu_action = [act, chosen_mode]
    return new_mode_list, tabu_action


def add2tabu_list(tabu_action, Tabu_list):
    tabu_action.append(tabu_tenure)
    Tabu_list.append(tabu_action)
    return Tabu_list


def update_tabu_list(Tabu_List_act, Tabu_List_mode):
    if len(Tabu_List_act) != 0:
        delet_list_act = []
        for x in range(len(Tabu_List_act)):
            Tabu_List_act[x][-1] -= 1
            if Tabu_List_act[x][-1] == 0:
                delet_list_act.append(Tabu_List_act[x])
        for d in delet_list_act:
            Tabu_List_act.remove(d)

    if len(Tabu_List_mode) != 0:
        delet_list_mode = []
        for x in range(len(Tabu_List_mode)):
            Tabu_List_mode[x][-1] -= 1
            if Tabu_List_mode[x][-1] == 0:
                delet_list_mode.append(Tabu_List_mode[x])
        for d in delet_list_mode:
            Tabu_List_mode.remove(d)

    return Tabu_List_act, Tabu_List_mode


def tabu_search(init_act_list, init_mode_list, init_total_cost):
    cur_solution = [init_act_list, init_mode_list]
    cur_obj = init_total_cost
    best_solution = copy.deepcopy(cur_solution)
    best_obj = copy.deepcopy(cur_obj)
    Tabu_List_act = []
    Tabu_List_mode = []
    num_sub = 0

    while num_sub <= max_sub:
        num_sub = num_sub + num_neighbors1 + num_neighbors2
        Neighbors1 = []
        Neighbors2 = []
        Tabu_Actions1 = []
        Tabu_Actions2 = []

        # ...generate neighbors...
        cur_act_list = cur_solution[0]
        cur_mode_list = cur_solution[1]

        # [act]
        for x in range(num_neighbors1):
            new_act_list, tabu_action = improved_swap2(cur_act_list)
            new_solution = [new_act_list, cur_mode_list]
            Neighbors1.append(new_solution)
            Tabu_Actions1.append(tabu_action)

        cur_duration_list = {j: Duration[j][cur_mode_list[j]] for j in range(1, task_num + 1)}
        ST, FT = ssgs_decoding(successors, predecessors, cur_act_list, cur_duration_list, resource_demand, s, S,
                               resource_capacity, task_num, num_r)
        # [mode]
        for x in range(num_neighbors1):
            new_mode_list, tabu_action = optimize_modes(cur_mode_list)
            new_solution = [cur_act_list, new_mode_list]
            Neighbors2.append(new_solution)
            Tabu_Actions2.append(tabu_action)

        Tabu_List_act, Tabu_List_mode = update_tabu_list(Tabu_List_act, Tabu_List_mode)
        # ...evaluate neighbors...
        ALL = []
        for x in range(0, len(Neighbors1)):
            ALL.append([Neighbors1[x], Tabu_Actions1[x], None, 'act'])
        for y in range(0, len(Neighbors2)):
            ALL.append([Neighbors2[y], Tabu_Actions2[y], None, 'mode'])

        # [act]
        for x in range(0, len(ALL)):
            new_solu = ALL[x][0]
            total_cost = get_total_cost(new_solu[0], new_solu[1])
            ALL[x][2] = total_cost

        sorted_ALL = sorted(ALL, key=lambda x: x[2])
        for x in range(len(sorted_ALL)):
            neighbor = sorted_ALL[x][0]
            move = sorted_ALL[x][1]
            neig_obj = sorted_ALL[x][2]
            origin = sorted_ALL[x][3]

            if neig_obj < best_obj:
                best_obj = neig_obj
                best_solution = neighbor
                cur_solution = neighbor
                cur_obj = neig_obj

                # update tabu lists
                if any(Tabu_List_act[:2]) == move or any(Tabu_List_mode[:2]) == move:  # aspiration criterion
                    if origin == 'act':
                        for index, sublist in enumerate(Tabu_List_act):
                            if sublist[:2] == move:
                                Tabu_List_act.remove(index)
                                Tabu_List_act = add2tabu_list(move, Tabu_List_act)
                    else:
                        for index, sublist in enumerate(Tabu_List_mode):
                            if sublist[:2] == move:
                                Tabu_List_mode.remove(index)
                                Tabu_List_mode = add2tabu_list(move, Tabu_List_mode)
                else:
                    if origin == 'act':
                        Tabu_List_act = add2tabu_list(move, Tabu_List_act)
                    else:
                        Tabu_List_mode = add2tabu_list(move, Tabu_List_mode)
                break
            else:
                if any(Tabu_List_act[:2]) != move and any(Tabu_List_mode[:2]) != move:
                    cur_obj = neig_obj
                    cur_solution = neighbor
                    if origin == 'act':
                        Tabu_List_act = add2tabu_list(move, Tabu_List_act)
                    else:
                        Tabu_List_mode = add2tabu_list(move, Tabu_List_mode)
                    break
        ALL_best_obj.append(best_obj)
    return best_obj, best_solution


if __name__ == '__main__':

    ALL = []
    # -----pars tabu-----
    max_sub = 10000
    # max_iterations = 100
    tabu_tenure = 3
    num_neighbors1 = 3
    num_neighbors2 = 3

    # -----paths-------
    Inst_Set = 'carbon-j10-sample'
    # Inst_Set = 'single_instance'
    path = 'E:/workspace/project scheduling/CarbonRCSPS Instances/sample/'
    Inst_Set_path = os.path.join(path, Inst_Set)
    Inst_List = os.listdir(Inst_Set_path)

    # ---------parameters of project--------
    # ...old par...
    v_slower_rate = 0.8  # low speed rate
    v_higher_rate = 1.2  # high speed rate
    theta = 3.06  # Factor for conversion of energy consumption to carbon emissions 3.06
    resource_price = [3, 4, 4, 5]  # prices of resources
    energy_price = 0.2  # price of energy
    Re = 25.0  # Reward for early completion
    Pe = 50.0  # Penalty for delayed completion
    # carbon_price = 0.05  # price of carbon trading

    # ...parameters of ladder carbon price...
    init_carbon_price = 0.05
    gap = 0.1
    p_inc = init_carbon_price / 10
    print('max_sub, tabu_tenure, num_neighbors')
    print(max_sub, tabu_tenure, num_neighbors1)
    print('instance', 'totalcost', 'time')
    # All_result = []
    # All_time = []
    # for inst in Inst_List:
    for u in range(0, 1):
        ALL_best_obj = []
        inst = Inst_List[u]
        inst_path = os.path.join(Inst_Set_path, inst)
        # -----read instance data---------
        task_num, num_r, successors, resource_demand, resource_capacity, wight_speed, quality_basic, wight_quality, \
        s, S, resource_speed_basic, work_load, delta, duration_basic, D = read_carbon_rcpsp_file(inst_path)

        # ------calculate other data------
        predecessors = Ca.get_pred(task_num, successors)
        Resource_Speed = Ca.get_resource_speed(num_r, v_slower_rate, v_higher_rate, resource_speed_basic)
        Activity_Speed = Ca.get_activity_speed(task_num, num_r, Resource_Speed, wight_speed, resource_demand)  # Calculate the speed of each mode for every activity
        Duration = Ca.get_durations(task_num, duration_basic, Activity_Speed)  # Calculate the duration of each mode for every activity
        Energy_Consumption = Ca.get_all_energy_consumption(Duration, Resource_Speed, delta, task_num, num_r, resource_demand)  # Calculate the energy consumption of each mode for every activity
        Quality = Ca.get_quality(task_num, Activity_Speed, quality_basic)  # Calculate the quality of each mode for every activity
        # Q = sum([quality_basic[key] * wight_quality[key] for key in quality_basic.keys() & wight_quality.keys()]) * 1.1
        Q = -1  # Ignore quality constrain
        Acts_space = Ca.get_space(s, resource_demand)  # space demand for executing every activity
        Quota = 0  # Carbon Quota
        for j in range(1, task_num + 1):
            Quota += round(Energy_Consumption[j][1] * theta * 0.8)

        start_t = time.time()
        init_act_list, init_mode_list, _ = encoding(task_num, successors, predecessors, Duration)
        # print(init_act_list, init_mode_list)
        init_total_cost = get_total_cost(init_act_list, init_mode_list)
        # print('MDDQN', inst, MDDQN_total_cost)
        """
        ++++++++++++++++++++++++++
        ++      tabu search     ++
        ++++++++++++++++++++++++++
        """
        Tabu_List_act = []
        Tabu_List_mode = []
        best_obj, _ = tabu_search(init_act_list, init_mode_list, init_total_cost)
        end_t = time.time()
        time_consuming = end_t-start_t
        print(inst, best_obj, time_consuming)
        ALL.append([inst, ALL_best_obj])
    for x in ALL:
        print(x)






