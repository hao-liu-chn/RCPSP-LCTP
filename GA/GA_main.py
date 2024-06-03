#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/17
# @Author  : name
# @File    : GA_main.py
# @Software : PyCharm

import copy

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from read_carbonRCPSP_data3 import read_carbon_rcpsp_file
from tqdm import tqdm
import os
import Caculate as Ca
import GA_box as GA
from ladder_carbon_price import *
import random

inst_set = 'carbon-j10-sample'
path_inst_set = 'E:/workspace/project scheduling/CarbonRCSPS Instances/sample/' + inst_set
Inst_List = os.listdir(path_inst_set)

# print(Chosen_Inst)
ALL = []


# _____pars of GA_____
popsize = 50
max_sch = 10000
cro_prob = 0.8
muta_prob = 0.3

# ---------data of project--------
# -----old par---------
v_slower_rate = 0.8  # low speed rate
v_higher_rate = 1.2  # high speed rate
theta = 3.06  # Factor for conversion of energy consumption to carbon emissions
resource_price = [3, 4, 4, 5]  # prices of resources
energy_price = 0.2  # price of energy
Re = 25.0  # Reward for early completion
Pe = 50.0  # Penalty for delayed completion
init_carbon_price = 0.05
gap = 0.1
p_inc = init_carbon_price/10


print('instance', 'time', 'result', 'checkQ')

All_result = []
for u in range(0, 1):
# for u in range(len(Inst_List)):
    ALL_best_obj = []
    start_t = time.time()
    # -------read data------
    filename = Inst_List[u]
    file_path = os.path.join(path_inst_set, filename)
    task_num, num_r, successors, resource_demand, resource_capacity, wight_speed, quality_basic, wight_quality, \
    s, S, resource_speed_basic, work_load, delta, duration_basic, D = read_carbon_rcpsp_file(file_path)
    predecessors = Ca.get_pred(task_num, successors)

    # ------calculate other data------
    Resource_Speed = Ca.get_resource_speed(num_r, v_slower_rate, v_higher_rate, resource_speed_basic)
    Activity_Speed = Ca.get_activity_speed(task_num, num_r, Resource_Speed, wight_speed,
                                           resource_demand)  # Calculate the speed of each mode for every activity
    Duration = Ca.get_durations(task_num, duration_basic,
                                Activity_Speed)  # Calculate the duration of each mode for every activity
    Energy_Consumption = Ca.get_all_energy_consumption(Duration, Resource_Speed, delta, task_num, num_r,
                                                       resource_demand)  # Calculate the energy consumption of each mode for every activity
    Quality = Ca.get_quality(task_num, Activity_Speed,
                             quality_basic)  # Calculate the quality of each mode for every activity
    # Q = sum([quality_basic[key] * wight_quality[key] for key in quality_basic.keys() & wight_quality.keys()]) * 1.1
    Q = -1  # Ignore quality constrain
    Acts_space = Ca.get_space(s, resource_demand)  # space demand for executing every activity
    Quota = 0  # Carbon Quota
    for j in range(1, task_num + 1):
        Quota += round(Energy_Consumption[j][1] * theta * 0.8)
    # print('Quota', Quota)

    # _____Main Loop_____
    num_sch = 0
    Avr_Obj = []
    Min_Total_Cost = []
    min_tc = None
    best_individual = None
    pop = GA.initialize(popsize, task_num, successors, predecessors, Duration, Quality, Q, Energy_Consumption, theta, wight_quality)
    while num_sch <= max_sch:
        num_sch += popsize
        decoded_pop = GA.ssgs_decoding(successors, predecessors, pop, resource_demand, s, S, resource_capacity, task_num, num_r)
        fitted_pop = GA.total_cost_fun(task_num, num_r, resource_demand, resource_price, energy_price, Re, Pe, Quota, pop, D, theta, gap, p_inc, init_carbon_price)
        # fitted_pop = GA.total_cost_fun(task_num, num_r, resource_demand, resource_price, energy_price, Re, Pe, carbon_price, Quota, pop, D, theta, Quality, wight_quality, Q)
        min_tc, best_individual = GA.find_min_value(fitted_pop, 'total_cost')
        avr_obj = GA.average_obj_value(fitted_pop)
        Avr_Obj.append(avr_obj)
        Min_Total_Cost.append(min_tc)
        selected_pop = GA.select(fitted_pop)
        cross_pop = GA.crossover(selected_pop, cro_prob, task_num, Quality, wight_quality, Q, Duration, Energy_Consumption, theta)
        muta_pop = GA.mutation(task_num, cross_pop, muta_prob, predecessors, successors, Quality, Q, wight_quality, Duration, Energy_Consumption, theta)
        pop = copy.deepcopy(muta_pop)
        ALL_best_obj.append(min_tc)

    # _____result_____
    result = min_tc
    ALL.append([filename, ALL_best_obj])

    quality_dict = {}
    for j in range(1, task_num + 1):
        j_mode = best_individual['mode'][j]
        quality_dict[j] = Quality[j][j_mode]
    project_quality = sum([quality_dict[key] * wight_quality[key] for key in quality_dict.keys() & wight_quality.keys()])
    if project_quality >= Q:
        checkQ = 'ok'
    else:
        checkQ = 'wrong'
    end_t = time.time()
    spend_time = end_t - start_t
    # print('--------------------------------')
    print(filename, spend_time, result, checkQ)
    # All_result.append(result)
# print('avr', sum(All_result)/len(All_result))
for x in ALL:
    print(x)