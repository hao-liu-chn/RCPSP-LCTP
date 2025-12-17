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


def sample_training_instance(extend_inst):
    # ------sample a training instance-----
    training_set_path = os.path.join(path, training_set)
    Training_Set = os.listdir(training_set_path)
    training_inst = random.choice(Training_Set)
    training_inst_path = os.path.join(training_set_path, training_inst)

    # -----read instance data---------
    task_num, num_r, successors, resource_demand, resource_capacity, wight_speed, quality_basic, wight_quality, \
    s, S, resource_speed_basic, work_load, delta, duration_basic, D = read_carbon_rcpsp_file(training_inst_path)
    

    # ------calculate other data------
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

    # ----extend training set------
    if extend_inst:
        for j in range(1, task_num+1):
            duration_rate = random.uniform(0.75, 1.25)
            Duration[j] = [round(Duration[j][m] * duration_rate) for m in range(3)]

    # -----calculate some other data-----
    init_total_cost = -(D * Re + Quota * init_carbon_price)
    max_makespan = sum([Duration[j][0] for j in range(1, task_num)])
    max_carbon = sum([Energy_Consumption[j][2] * theta for j in range(1, task_num)])

    # ------create an environment-----
    env = EnvCRCPSP(task_num, successors, resource_capacity, resource_demand, S, Acts_space, Duration,
                    init_total_cost, theta, resource_price, energy_price, Re, Pe, init_carbon_price, gap, p_inc,
                    Energy_Consumption, Quota, D)
    return training_inst, env, max_makespan, max_carbon, task_num


def test_on_testing_set():
    Total_Cost_test = []
    # ------load a testing instance-----
    testing_set_path = os.path.join(path, testing_set)
    Testing_Set = os.listdir(testing_set_path)
    for testing_inst in Testing_Set:
        testing_inst_path = os.path.join(testing_set_path, testing_inst)
        # -----read instance data---------
        task_num, num_r, successors, resource_demand, resource_capacity, wight_speed, quality_basic, wight_quality, \
        s, S, resource_speed_basic, work_load, delta, duration_basic, D = read_carbon_rcpsp_file(testing_inst_path)

        # ------calculate other data------
        Resource_Speed = Ca.get_resource_speed(num_r, v_slower_rate, v_higher_rate, resource_speed_basic)
        Activity_Speed = Ca.get_activity_speed(task_num, num_r, Resource_Speed, wight_speed, resource_demand)  # Calculate the speed of each mode for every activity
        Duration = Ca.get_durations(task_num, duration_basic, Activity_Speed)  # Calculate the duration of each mode for every activity
        Energy_Consumption = Ca.get_all_energy_consumption(Duration, Resource_Speed, delta, task_num, num_r, resource_demand)  # Calculate the energy consumption of each mode for every activity
        Quality = Ca.get_quality(task_num, Activity_Speed, quality_basic)  # Calculate the quality of each mode for every activity （this line is useless!）
        # Q = sum([quality_basic[key] * wight_quality[key] for key in quality_basic.keys() & wight_quality.keys()]) * 1.1
        Q = -1  # Ignore quality constrain
        Acts_space = Ca.get_space(s, resource_demand)  # space demand for executing every activity
        Quota = 0  # Carbon Quota
        for j in range(1, task_num + 1):
            Quota += round(Energy_Consumption[j][1] * theta * 0.8)
        init_total_cost = -(D * Re + Quota * init_carbon_price)
        max_makespan_test = sum([Duration[j][0] for j in range(1, task_num)])
        max_carbon_test = sum([Energy_Consumption[j][2] * theta for j in range(1, task_num)])

        # ------create an environment-----
        env_test = EnvCRCPSP(task_num, successors, resource_capacity, resource_demand, S, Acts_space, Duration,
                             init_total_cost, theta, resource_price, energy_price, Re, Pe, init_carbon_price, gap, p_inc,
                             Energy_Consumption, Quota, D)

        # ------solve the testing instance by Agent------
        RL_DQN.max_carbon = max_carbon_test
        RL_DQN.max_makespan = max_makespan_test
        obs_test = env_test.reset()
        done_test = False
        while not done_test:
            Act_Set_test = env_test.get_actions_set(obs_test)
            act_test = RL_DQN.choose_action(obs_test, Act_Set_test)
            new_obs_test, _, done_test = env_test.step(act_test)
            obs_test = new_obs_test

        # -------calculate total cost of the testing instance---------
        total_cost_test, _, _, _, _ = env_test.get_total_cost()
        Total_Cost_test.append(total_cost_test)
    avr_total_cost_test = sum(Total_Cost_test)/len(Total_Cost_test)
    # ----Change back to the parameters used by the Agent for training----
    RL_DQN.max_carbon = max_carbon
    RL_DQN.max_makespan = max_makespan
    return avr_total_cost_test


if __name__ == '__main__':
    LR = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    E_decay = [1/1200, 1/1500, 1/2000, 1/2500]
    auto_num = 0
    for lr in LR:
        for epsilon_decay_rate in E_decay:
            auto_num += 1
            start_t = time.time()
            # -----paths-------
            pars = str(auto_num) + '_lr' + str(lr) + '_ed_1_' + str(1 / epsilon_decay_rate) + '_re1'
            print('---------', pars, '-----------')
            log_dir = 'logsold_auto_ladder_re1/' + pars
            training_set = 'carbon-j10-train'
            testing_set = 'carbon-j10-test'
            # training_set = 'single_instance'
            # testing_set = 'single_instance'
            path = 'E:/workspace/project scheduling/CarbonRCSPS Instances/'  # local path
            # path = '/apps/users/raleigh_ncsu/liuhao/CarbonRCSPS Instances/'  # online path
            model_save_path = './trained model/' + pars + '/trained_model'

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

            # -----parameters of ladder carbon price----
            init_carbon_price = 0.05
            gap = 0.1
            p_inc = init_carbon_price / 10

            # --------parameters of Agent------------
            # ...parameters of CNN...
            num_filters = 5
            kernel_size = (4, 1)
            features_dim = 7  # dimension of obs1 + dimension of action ([C, A, M, ST, AC] + [acts, speeds])
            num_indicators = 5
            # ...parameters of RL...
            multi_step = 2
            double_dqn = True
            extend_inst = False
            change_lr = False
            EPISODES = 3000
            epsilon_start = 0.95
            epsilon_end = 0.00001
            # epsilon_decay_rate = 1 / 2000  # 1/1200
            # lr = 5e-4  # 1e-4
            lr_min = 1e-4
            # warmth_rate = 0.1
            disc = 1
            rep_tar = 50  # every fixed steps to replace target network wight
            mem_size = 1000
            batch_size = 64
            loss_threshold = 0.0001
            inst_change_gap = 1  # change training instance
            # ------create an initial env-----
            training_inst, env, max_makespan, max_carbon, task_num = sample_training_instance(extend_inst)

            # ------create DQN Agent-----
            RL_DQN = DQN(num_tasks=task_num,
                         learning_rate=lr,
                         discount=disc,
                         replace_target_iter=rep_tar,
                         memory_size=mem_size,
                         batch_size=batch_size,
                         epsilon_start=epsilon_start,
                         epsilon_end=epsilon_end,
                         epsilon_decay_rate=epsilon_decay_rate,
                         loss_threshold=loss_threshold,
                         num_filters=num_filters,
                         num_indicators=num_indicators,
                         kernel_size=kernel_size,
                         features_dim=features_dim,
                         double_dqn=double_dqn,
                         max_makespan=max_makespan,
                         max_carbon=max_carbon)

            # -----Cosine Annealing LR Scheduler------
            # Lr_scheduler = CosineAnnealingLRScheduler(EPISODES, lr, lr_min, warmth_rate)
            Lr_scheduler = CosineAnnealingLRScheduler3(500, lr, lr_min)

            # -------Training-------
            loss = None
            training_step = 0
            Record_Total_Reward = []
            for episode in range(1, EPISODES+1):
                if RL_DQN.stop_training:
                    print('Meet the conditions to stop training')
                    break

                # ----change learning rate------
                if change_lr:
                    RL_DQN.lr = Lr_scheduler(episode)
                    with tf.summary.create_file_writer(log_dir).as_default():
                        tf.summary.scalar('lr-episode', RL_DQN.lr, step=episode)

                # if episode % 10 == 0:
                #     print(f'episode={episode},instance={training_inst}, '
                #           f'loss={loss},epsilon={RL_DQN.epsilon}, lr={RL_DQN.lr}')

                total_reward = 0
                exp = []  # an experience. observation = [[[C], [A], [M], [ST], [Indicators]]; exp = [obs_i, action_i, r_i, obs_i+1, ...,obs_i+n, steps]
                m_steps = 0

                obs = env.reset()
                done = False
                while not done:
                    Act_Set = env.get_actions_set(obs)
                    action = RL_DQN.choose_action(obs, Act_Set)
                    new_obs, reward, done = env.step(action)

                    m_steps += 1
                    RL_DQN.count_step += 1
                    RL_DQN.update_epsilon()
                    total_reward += reward  # useless when train on sample
                    exp += [obs, action, reward]  # update exp
                    obs = new_obs  # update observation/state

                    # ------restore a new experience------
                    if m_steps % multi_step == 0:
                        exp.append(new_obs)
                        exp.append(m_steps)
                        RL_DQN.store_transition(exp)
                        exp = []
                        m_steps = 0

                    # ------Extract experience and train on batch-----
                    if (RL_DQN.count_step > 200) and (RL_DQN.count_step % 5 == 0) and not RL_DQN.stop_training:
                        Batch_exp = RL_DQN.get_batch()
                        Actions_set_of_Batch_last_obs = [env.get_actions_set(Batch_exp[x][-2]) for x in
                                                         range(len(Batch_exp))]
                        loss = RL_DQN.learn(Batch_exp, Actions_set_of_Batch_last_obs)
                        training_step += 1
                        # ........record logs for tensorboard.......
                        with tf.summary.create_file_writer(log_dir).as_default():
                            tf.summary.scalar('loss', loss, step=training_step)
                            tf.summary.scalar('epsilon', RL_DQN.epsilon, step=RL_DQN.count_step)
                        if len(RL_DQN.loss_his) > 500:
                            RL_DQN.check_stop_training()

                # -----after an episode is completed------
                Record_Total_Reward.append(total_reward)  # useless when train on sample
                # with tf.summary.create_file_writer(log_dir + '/tr').as_default():
                #     tf.summary.scalar('total_reward', total_reward, step=episode)
                if done and 0 < m_steps < multi_step:  # sometimes the length of exp < multi_step but the episode is done
                    exp.append(new_obs)
                    exp.append(m_steps)
                    RL_DQN.store_transition(exp)

                # -----Testing-----------
                if episode % 20 == 0:
                    avr_total_cost_test = test_on_testing_set()
                    # print('avr_total_cost_test', avr_total_cost_test)
                    with tf.summary.create_file_writer(log_dir).as_default():
                        tf.summary.scalar('TestingSet avr total cost', avr_total_cost_test, step=episode)

                # ------sample a new instance for training-------
                if episode % inst_change_gap == 0:
                    training_inst, env, max_makespan, max_carbon, task_num = sample_training_instance(extend_inst)
                    RL_DQN.max_carbon = max_carbon
                    RL_DQN.max_makespan = max_makespan

            # ------save model----
            RL_DQN.save_model(model_save_path)

            # -----output-----
            finish_t = time.time()
            training_time = finish_t - start_t

            print('任务', auto_num, 'training_time', training_time)

