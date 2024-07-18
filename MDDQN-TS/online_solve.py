import copy
import random

from env import EnvCRCPSP
from Agent import DQN
import numpy as np
import time
import matplotlib.pyplot as plt
from read_carbonRCPSP_data3 import read_carbon_rcpsp_file
from tqdm import tqdm
import os
import Caculate as Ca
# from MoreStates import *
import tensorflow as tf
from LR_scheduler import *
from ladder_carbon_price import *
from SSGS_decoding import ssgs_decoding
from total_float import calculate_total_float
import warnings


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


def improved_swap2(act_list, Tabu_List_act):
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
            tabu = check_tabu_list_act(act1, act2, Tabu_List_act)
            if not tabu:  
                new_act_list[pos1] = act2
                new_act_list[pos2] = act1
                if new_act_list != act_list:
                    same = False
                    tabu_action = [act1, act2]
            else:
                # print('被禁')
                pass
        if count > 10:
            break

    # if new_act_list == act_list:
    #     new_act_list = None
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
        stop = 10
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
                # print('被禁mode')
        # if best_cost < old_cost:
        #     print('good')
        # else:
        #     print('not useful')
    return new_mode_list, tabu_action


def optimize_modes(mode_list, Tabu_List_mode):
    new_mode_list = copy.deepcopy(mode_list)
    tabu_action = [None, None]
    U = list(range(1, task_num))
    stop = 10
    count = 1
    while count <= stop:
        count += 1
        act = random.choice(U)
        original_mode = mode_list[act]
        modes_for_choose = [i for i in range(0, 3) if i != original_mode]
        chosen_mode = random.choice(modes_for_choose)
        tabu = check_tabu_list_mode(act, chosen_mode, Tabu_List_mode)
        if not tabu:
            new_mode_list[act] = chosen_mode
            tabu_action = [act, chosen_mode]
            break
        else:
            pass
            # print('被禁mode')
 
    return new_mode_list, tabu_action


def add2tabu_list(tabu_action, Tabu_list):
    tabu_action.append(tabu_tenure)
    if len(Tabu_list) < tabu_size:
        Tabu_list.append(tabu_action)
    else:
        Tabu_list.pop(0)
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


def tabu_search(init_act_list, init_mode_list, init_total_cost, cp=False):
    cur_solution = [init_act_list, init_mode_list]
    cur_obj = init_total_cost
    best_solution = copy.deepcopy(cur_solution)
    best_obj = copy.deepcopy(cur_obj)
    Tabu_List_act = []
    Tabu_List_mode = []

    for ite in range(max_iterations):
        Neighbors1 = []
        Neighbors2 = []
        Tabu_Actions1 = []
        Tabu_Actions2 = []

        # ...generate neighbors...
        cur_act_list = cur_solution[0]
        cur_mode_list = cur_solution[1]

        for x in range(num_neighbors1):
            new_act_list, tabu_action = improved_swap2(cur_act_list, Tabu_List_act)
            new_solution = [new_act_list, cur_mode_list]
            Neighbors1.append(new_solution)
            Tabu_Actions1.append(tabu_action)

        cur_duration_list = {j: Duration[j][cur_mode_list[j]] for j in range(1, task_num + 1)}
        ST, FT = ssgs_decoding(successors, predecessors, cur_act_list, cur_duration_list, resource_demand, s, S,
                               resource_capacity, task_num, num_r)

        for x in range(num_neighbors1):
            if cp:
                new_mode_list, tabu_action = optimize_modes_cp(cur_mode_list, ST, FT, Tabu_List_mode)
                new_solution = [cur_act_list, new_mode_list]
                Neighbors2.append(new_solution)
                Tabu_Actions2.append(tabu_action)
            else:
                new_mode_list, tabu_action = optimize_modes(cur_mode_list, Tabu_List_mode)
                new_solution = [cur_act_list, new_mode_list]
                Neighbors2.append(new_solution)
                Tabu_Actions2.append(tabu_action)

        Neighbors1_Obj = []
        for new_solu in Neighbors1:
            total_cost = get_total_cost(new_solu[0], new_solu[1])
            Neighbors1_Obj.append(total_cost)
            best_obj_Neighbors1 = min(Neighbors1_Obj)
            index = Neighbors1_Obj.index(best_obj_Neighbors1)
            best_solution_Neighbors1 = Neighbors1[index]
            best_tabu_action1 = Tabu_Actions1[index]

        Neighbors2_Obj = []
        for new_solu in Neighbors2:
            total_cost = get_total_cost(new_solu[0], new_solu[1])
            Neighbors2_Obj.append(total_cost)
            best_obj_Neighbors2 = min(Neighbors2_Obj)
            index = Neighbors2_Obj.index(best_obj_Neighbors2)
            best_solution_Neighbors2 = Neighbors2[index]
            best_tabu_action2 = Tabu_Actions2[index]

        Tabu_List_act, Tabu_List_mode = update_tabu_list(Tabu_List_act, Tabu_List_mode)
        if best_obj_Neighbors1 < best_obj_Neighbors2:
            cur_solution = best_solution_Neighbors1
            cur_obj = best_obj_Neighbors1
            Tabu_List_act = add2tabu_list(best_tabu_action1, Tabu_List_act)
        else:
            cur_solution = best_solution_Neighbors2
            cur_obj = best_obj_Neighbors2
            Tabu_List_mode = add2tabu_list(best_tabu_action2, Tabu_List_mode)

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_solution = cur_solution
        # print(best_obj)
        # print('Tabu_List_mode', Tabu_List_mode)
        # print('Tabu_List_act', Tabu_List_act)
    return best_obj, best_solution


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tabuSearch = False
    max_iterations = 100
    tabu_tenure = 3
    tabu_size = 10
    num_neighbors1 = 3
    num_neighbors2 = 3

    Inst_Set = 'carbon-j120-sample'
    path = 'E:/workspace/project scheduling/CarbonRCSPS Instances/sample/'  

    Inst_Set_path = os.path.join(path, Inst_Set)
    Inst_List = os.listdir(Inst_Set_path)

    v_slower_rate = 0.8  # low speed rate
    v_higher_rate = 1.2  # high speed rate
    theta = 3.06  # Factor for conversion of energy consumption to carbon emissions 3.06
    resource_price = [3, 4, 4, 5]  # prices of resources
    energy_price = 0.2  # price of energy
    Re = 25.0  # Reward for early completion
    Pe = 50.0  # Penalty for delayed completion
    # carbon_price = 0.05  # price of carbon trading

    init_carbon_price = 0.05
    gap = 0.1
    p_inc = init_carbon_price / 10

    num_filters = 5
    kernel_size = (4, 1)
    features_dim = 7  # dimension of obs1 + dimension of action ([C, A, M, ST, AC] + [acts, speeds])
    num_indicators = 5

    All_result = []
    All_time = []
    for inst in tqdm(Inst_List):
        inst = 'j1209_6.txt'

        inst_path = os.path.join(Inst_Set_path, inst)
        task_num, num_r, successors, resource_demand, resource_capacity, wight_speed, quality_basic, wight_quality, \
        s, S, resource_speed_basic, work_load, delta, duration_basic, D = read_carbon_rcpsp_file(inst_path)

        predecessors = Ca.get_pred(task_num, successors)
        Resource_Speed = Ca.get_resource_speed(num_r, v_slower_rate, v_higher_rate, resource_speed_basic)
        Activity_Speed = Ca.get_activity_speed(task_num, num_r, Resource_Speed, wight_speed, resource_demand)  # Calculate the speed of each mode for every activity
        Duration = Ca.get_durations(task_num, duration_basic, Activity_Speed)  # Calculate the duration of each mode for every activity
        Energy_Consumption = Ca.get_all_energy_consumption(Duration, Resource_Speed, delta, task_num, num_r, resource_demand)  # Calculate the energy consumption of each mode for every activity
        Quality = Ca.get_quality(task_num, Activity_Speed, quality_basic)  # Calculate the quality of each mode for every activity （useless!）
        # Q = sum([quality_basic[key] * wight_quality[key] for key in quality_basic.keys() & wight_quality.keys()]) * 1.1
        Q = -1  # Ignore quality constrain
        Acts_space = Ca.get_space(s, resource_demand)  # space demand for executing every activity
        Quota = 0  # Carbon Quota
        for j in range(1, task_num + 1):
            Quota += round(Energy_Consumption[j][1] * theta * 0.8)
        init_total_cost = 1  # useless in online solving

        # ------create DQN Agent-----
        RL_DQN = DQN(num_tasks=task_num,
                     learning_rate=1,
                     discount=1,
                     replace_target_iter=1,
                     memory_size=1,
                     batch_size=1,
                     epsilon_start=1,
                     epsilon_end=1,
                     epsilon_decay_rate=1,
                     loss_threshold=1,
                     num_filters=num_filters,
                     num_indicators=num_indicators,
                     kernel_size=kernel_size,
                     features_dim=features_dim,
                     double_dqn=True,
                     max_makespan=1,
                     max_carbon=1)
        # ------------load model------------
        model_path = '.\\trained model\\'
        model = 'lr=1e-05ed=1_1500ms=2mem=1000'
        save_path = os.path.join(model_path, model, 'model')
        RL_DQN.load_model(save_path)

        # ------create an environment-----
        env = EnvCRCPSP(task_num, successors, resource_capacity, resource_demand, S, Acts_space, Duration,
                        init_total_cost, theta, resource_price, energy_price, Re, Pe, init_carbon_price, gap, p_inc,
                        Energy_Consumption, Quota, D)

        """
        ++++++++++++++++++++++++++++++++++++++++++++++
        +               ONLINE SOLVE                 +
        ++++++++++++++++++++++++++++++++++++++++++++++
        """
        start_t = time.time()
        act_list = [1]
        mode_list = {j: None for j in range(1, task_num + 1)}
        mode_list[1] = 0

        obs = env.reset()
        done = False
        while not done:
            Act_Set = env.get_actions_set(obs)
            action = RL_DQN.choose_action_max(obs, Act_Set)
            new_obs, reward, done = env.step(action)
            obs = new_obs  # update observation/state
            act_list += action[0]
            for x in range(len(action[1])):
                act = action[0][x]
                mode_list[act] = action[1][x]

        MDDQN_total_cost = get_total_cost(act_list, mode_list)
        # print('MDDQN', inst, MDDQN_total_cost)
        """
        ++++++++++++++++++++++++++
        +       tabu search      +
        ++++++++++++++++++++++++++
        """
        if tabuSearch:
            Tabu_List_act = []
            Tabu_List_mode = []
            tabu_total_cost, _ = tabu_search(act_list, mode_list, MDDQN_total_cost, cp=False)
            end_t = time.time()
            time_consuming = end_t-start_t
            print('MDDQN+tabuS', inst, tabu_total_cost, time_consuming)
            # All_result.append(tabu_total_cost)
            # All_time.append(time_consuming)
        else:
            end_t = time.time()
            time_consuming = end_t - start_t
            print('MDDQN', inst, MDDQN_total_cost, time_consuming)
            # All_result.append(MDDQN_total_cost)
            # All_time.append(time_consuming)

    # print('----------------------')
    # print('avr_totalCost', sum(All_result)/len(All_result))
    # print('avr_time', sum(All_time)/len(All_time))



