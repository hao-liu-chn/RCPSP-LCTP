import random
import numpy as np
import math
import pandas as pd
import copy
import operator


# 推算紧前关系
def get_pred(tasks_num_, successors_):
    predecessors_ = {i: [] for i in range(1, tasks_num_ + 1)}
    for task in range(1, tasks_num_):
        succ = successors_[task]
        for i in succ:
            predecessors_[i].append(task)
    return predecessors_


# 推算不同模式资源工作速度
def get_resource_speed(num_r, v_slower_rate, v_higher_rate, resource_speed_basic):
    Resource_Speed = []
    for k in range(num_r):
        Resource_Speed.append([round(resource_speed_basic[k] * v_slower_rate), resource_speed_basic[k],
                               round(resource_speed_basic[k] * v_higher_rate)])
    return Resource_Speed


# 推算不同模式活动执行速度
def get_activity_speed(act_num, num_r, Resource_Speed, wight_speed, r):
    Activity_Speed = {}
    for j in range(1, act_num + 1):
        temp_speed = []
        for x in range(3):
            v = 0
            for k in range(num_r):
                v += wight_speed[j][k] * r[j][k] * Resource_Speed[k][x]
            temp_speed.append(round(v))
        Activity_Speed[j] = temp_speed
    return Activity_Speed


# 推算不同模式下的活动工期
def get_durations(act_num, duration_basic, Activity_Speed):  # 各个模式下的活动工期
    J = act_num
    Duration = {}
    for j in range(1, J + 1):
        if j == 1 or j == act_num:
            Duration[j] = [0, 0, 0]
        else:
            temp = []
            for x in range(3):
                temp.append(round(duration_basic[j] * Activity_Speed[j][1] / Activity_Speed[j][x]))
            Duration[j] = temp
    return Duration


# 推算不同模式下的活动能耗
def get_all_energy_consumption(Duration, Resource_Speed, delta, act_num, num_r, r):  # 输出每个活动每种模式下的能耗量
    J = act_num
    Energy_Consumption = {}  # 记录每种活动的全部模式下的能耗
    for j in range(1, J + 1):
        if j == 1 or j == J:
            Energy_Consumption[j] = [0, 0, 0]
        else:
            temp = []
            for x in range(3):  # 3种等级的速度
                temp_e = 0
                for k in range(num_r):
                    temp_e += round(delta[k] * Resource_Speed[k][x] * Resource_Speed[k][x] * Resource_Speed[k][x] * r[j][k] * Duration[j][x])
                temp.append(temp_e)
            Energy_Consumption[j] = temp
    return Energy_Consumption


# 推算不同模式下的活动质量
def get_quality(act_num, Activity_Speed, quality_basic):
    Quality = {}
    for j in range(1, act_num + 1):
        if j == 1 or j == act_num:
            Quality[j] = [0, 0, 0]
        else:
            temp = []
            for x in range(3):
                if x == 1:
                    temp.append(quality_basic[j])
                else:
                    temp.append(round(quality_basic[j] * Activity_Speed[j][1] / Activity_Speed[j][x]))
            Quality[j] = temp
    return Quality


def get_space(s, resource_demand):
    act_num = len(resource_demand)
    Acts_space = {}
    for j in range(1, act_num+1):
        act_s = sum(list(map(lambda x, y: x*y, s, resource_demand[j])))
        Acts_space[j] = act_s
    return Acts_space

