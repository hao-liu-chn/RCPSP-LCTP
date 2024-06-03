# CNN 提速
# CNN
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import math
from tensorflow.keras.optimizers import SGD

from Net import Network


class DQN:
    def __init__(
            self,
            num_tasks,  # 活动数量
            learning_rate,  # 学习率
            discount,  # 折扣率
            replace_target_iter,  # 每隔一定步数替换目标网络的权重
            memory_size,  # 经验储存库容量
            batch_size,  # 批量
            epsilon_start,  # ε最大值
            epsilon_end,  # ε最小值
            epsilon_decay_rate,  # 控制ε衰减
            loss_threshold,  # 停止训练的误差阈值
            num_filters,  # 卷积核数量
            kernel_size,  # 卷积核尺寸 元组
            features_dim,  # 特征矩阵维度(一部分状态+动作)
            num_indicators,  # 补充状态的指标数量
            double_dqn,  # 是否用double dqn
            max_makespan,  # 全部选择最慢速度的活动工期之和
            max_carbon  # 全部选择最快速度的活动碳排放之和
    ):
        self.count_step = 0
        self.num_tasks = num_tasks  # 活动数量
        self.lr = learning_rate  # 学习率
        self.discount = discount  # 折扣率
        self.replace_target_iter = replace_target_iter  # 每隔一定的步数替换目标网络的权重系数
        self.memory_size = memory_size  # 经验储存库的大小
        self.batch_size = batch_size  # 每批抽取经验的数量
        self.epsilon_start = epsilon_start  # epsilon最大值
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.learn_step_counter = 0  # 学习步数
        self.memory = {i: [] for i in range(self.memory_size)}  # 字典-储存经验
        self.epsilon = epsilon_start
        self.memory_counter = 0  # 记录产生的经验数量
        self.loss_threshold = loss_threshold
        self.stop_training = False
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.pool_len = self.num_tasks - self.kernel_size[0] + 1
        self.features_dim = features_dim
        self.num_indicators = num_indicators
        self.double_dqn = double_dqn
        self.max_makespan = max_makespan
        self.max_carbon = max_carbon
        self.model_eval = Network(self.num_tasks, self.features_dim, self.num_filters, self.kernel_size, self.num_indicators)
        self.model_target = Network(self.num_tasks, self.features_dim, self.num_filters, self.kernel_size, self.num_indicators)
        self.model_eval.compile(optimizer=RMSprop(lr=self.lr), loss='mse')  # 配置训练方法：选择哪种优化器、哪种损失函数，哪种评测指标
        # 记录mse(TD_error)
        self.loss_his = []  # 记录历史训练误差
        self.epsilon_his = []  # 记录历史epsilon

    def _replace_target_params(self):  # 替换目标网络的权重
        for eval_layer, target_layer in zip(self.model_eval.layers, self.model_target.layers):
            target_layer.set_weights(eval_layer.get_weights())

    def store_transition(self, exp):  # 储存经验
        index = self.memory_counter % self.memory_size  # 经验索引值，后续经验自动覆盖老的经验
        self.memory[index] = exp
        self.memory_counter += 1

    def get_batch(self):
        """
        随机批量抽取经验
        :return: Batch_exp
        """
        if self.memory_counter > self.memory_size:  # 经验储存库已满，从整个库中抽取一批经验
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:  # 如果经验储存库没满，但产生的经验数量已经大于一批经验量所需的数量，则从已产生的经验中抽取
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)  # 如果产生的经验数量小于batch size怎么办？？
        Batch_exp = []
        for i in sample_index:
            Batch_exp.append(self.memory[i])
        return Batch_exp

    def learn(self, Batch_exp, Actions_set_of_Batch_last_obs):  # 使用前先调用环境类中的get_actions_set把batch_s_对应的动作集合全部找出来
        """
        Double DQN：用评估网络选择s_下所有action中Q值最大的action，用目标网络计算（s_, action）的Q值，再加上奖励r作为训练评估网络target
        :param Batch_exp=[exp1, ..., expn];  exp=[(s_i), [a_i], r_i, (s_i+1), ..., (s_i+n), m_steps]
        :param actions_set_of_batch_s_: 全部状态s_各自对应的动作集合
        :return: 目标网络的训练误差
        """
        # print('exp\n', Batch_exp[0])
        # for b in range(self.batch_size):
        #     print('type_exp\n', type(Batch_exp[b]))
        st = time.time()
        Batch_obs1 = [Batch_exp[b][0][0] for b in range(self.batch_size)]  # [C,A,M,ST]
        Batch_obs2 = [Batch_exp[b][0][1] for b in range(self.batch_size)]  # [indicator_1, ...,indicator_n]
        Batch_action = [Batch_exp[x][1] for x in range(self.batch_size)]
        Batch_last_obs1 = [Batch_exp[b][-2][0] for b in range(self.batch_size)]
        Batch_last_obs2 = [Batch_exp[b][-2][1] for b in range(self.batch_size)]
        # =============构建评估网络的输入================
        # 将Batch_input_obs1_action_obs2转化成numpy数组格式，shape=(batch_size, num_tasks, features_dim+1, channels=1)
        # 其中features_dim+1的“+1”指的是把由补充指标组成的列向量加进来，列向量长度不足num_tasks的用0填充
        Batch_input_obs1_action_obs2 = self.preprocess_CNNinput_obs1_action_obs2(Batch_obs1, Batch_action, Batch_obs2)
        State_Action_eval = self.patch_obs2arry(Batch_input_obs1_action_obs2)
        # ===============构建目标网络的输入(如果是DDQN的话就输入到评估网络中)=================
        Batch_input_last_obs1_actionSet_obs2, Num_Actions = self.preprocess_CNNinput_last_obs1_actionSet_obs2(Batch_last_obs1, Actions_set_of_Batch_last_obs, Batch_last_obs2)
        State_Action_target = self.patch_obs2arry(Batch_input_last_obs1_actionSet_obs2)

        if self.double_dqn:
            Batch_last_q = self.model_eval.call(State_Action_target)  # DDQN
        else:
            Batch_last_q = self.model_target.call(State_Action_target)  # DQN
        Batch_last_q = np.ravel(Batch_last_q)  # 拉直为一维数组
        # --------------选择Q值最大的动作---------------
        Num_Actions.insert(0, 0)
        Batch_best_action = []
        start = 0
        for i in range(len(Num_Actions)-1):
            if Num_Actions[i+1] == 0:  # 该状态下对应的动作数量为0个，说明该状态时结束状态
                start += Num_Actions[i]
                temp_best_action = ([], [])  # s_为结束状态，最佳动作则为空
            else:
                start += Num_Actions[i]
                end = start + Num_Actions[i + 1]
                temp = Batch_last_q[start: end]  # 该状态下全部动作对应的Q值
                index = np.argmax(temp)  # Q值最大的动作的索引
                temp_best_action = Actions_set_of_Batch_last_obs[i][index]  # Q值最大的动作
            Batch_best_action.append(temp_best_action)  # 储存各个s_的最佳动作

        # --------------使用目标网络计算（s_,best_action）的Q值-----------------
        Batch_input_last_obs1_bestAction_obs2 = self.preprocess_CNNinput_obs1_action_obs2(Batch_last_obs1, Batch_best_action, Batch_last_obs2)
        LastState_BestAction = self.patch_obs2arry(Batch_input_last_obs1_bestAction_obs2)
        next_q = self.model_target.call(LastState_BestAction)  # DDQN-CRCPSP 用目标网络重新计算Q值
        # print('前向计算成功')
        next_q = next_q.numpy()  # 把输出的张量转化为数组
        next_q = np.ravel(next_q)  # 拉直
        Num_Actions.pop(0)
        Num_Actions = np.asarray(Num_Actions)
        index_end = np.argwhere(Num_Actions == 0)
        if len(index_end) != 0:  # 如果存在某个s_对应的动作集合是空集则为结束状态
            next_q[index_end] = 0  # 将结束时的状态对应的q值设置为0

        # 计算q_target，注意可能存在比正常经验要短的经验
        Q_target = []
        for b in range(self.batch_size):
            m_step = Batch_exp[b][-1]
            q_target = 0
            for m in range(m_step):
                q_target += self.discount ** m * Batch_exp[b][(m+1)*3-1]
            q_target += self.discount ** m_step * np.asarray(next_q[b])
            Q_target.append(q_target)
        Q_target = np.array(Q_target)

        # ==============训练评估网络===============
        ft = time.time()

        loss = self.model_eval.train_on_batch(State_Action_eval, Q_target)  # 输入是s,对应的输出是目标网络的输出，输出loss是损失值 对这个批次进行训练
        ft2 = time.time()
        # print('batch 处理时间', ft - st, 'learn', ft2 - ft)
        # ==============/其他附属步骤==============
        self.loss_his.append(loss)  # 记录loss
        self.learn_step_counter += 1  # 记录学习次数

        # ==========每隔若干步 替换目标网络的权重==========
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
        return loss

    def check_stop_training(self):  # 检查训练误差是否足够小，是否停止训练
        if np.mean(self.loss_his[-200:]) < self.loss_threshold:  # 当误差值均值变化不大或最后误差均值小于阈值时，停止训练神经网络
            self.stop_training = True
            self.epsilon = self.epsilon_end

    def choose_action(self, observation, actions_set):  # 依据当前状态，计算有多少动作，并计算每个状态动作对的Q值，从而选择下一步动作
        """
        使用贪婪策略
        在主程序中调用环境，先通过EnvRCPSP中的get_actions_set来识别有多少个动作,得到的actions_set的形式如actions_set=[(21, 22), (22, 27)]
        再将状态和动作组合成状态动作对，并输入评价网络，计算每个状态动作对的Q值,从中选择Q值最大的动作action_max
        再选择最大Q值对应的动作
        """
        # print('def choose_action\n', 'observation\n', observation, '\nactions_set\n', actions_set)
        if np.random.uniform() < 1-self.epsilon:  # 选择Q值最大的动作
            input_data, _ = self.preprocess_CNNinput_last_obs1_actionSet_obs2([observation[0]],
                                                                              [actions_set],
                                                                              [observation[1]])
            input_data = self.patch_obs2arry(input_data)
            # ----------使用评估网络计算Q值-------
            q_values = self.model_eval.call(input_data)
            # ---------选择Q值最大的动作---------
            q_values = np.ravel(q_values)  # 拉直为一维数组
            indexes = np.where(q_values == max(q_values))[0]  # where返回的是一个元组，如(array([0, 2], dtype=int64),)，将其中的数组剥离出来
            index = np.random.choice(indexes)  # 状态s下可能存在多个动作的Q值是相等的，从中任意选择一个
            best_action = actions_set[index]

        else:  # 随机选择动作（一开始要尽量多探索，然后逐渐收敛，选择Q值大的动作）
            index = np.random.choice(len(actions_set))
            best_action = actions_set[index]
        best_action = list(best_action)  # 要把元组形式转化为列表形式，以便后面操作
        return best_action

    def plot_loss(self):  # 画图
        moving_len = 100
        moving_avg = np.convolve(self.loss_his, np.ones((moving_len,)) / moving_len, mode='valid')

        # plt.plot([i for i in range(len(self.loss_his))], self.loss_his, label='loss')
        plt.plot([i for i in range(len(moving_avg))], moving_avg, label='loss moving average')
        plt.legend()
        plt.ylabel('mse')
        plt.xlabel('training_steps')
        plt.show()

    def save_model(self, filepath):
        self.model_eval.save_weights(filepath)

    def load_model(self, filepath):
        self.model_eval.load_weights(filepath)
        self.model_target.set_weights(self.model_eval.get_weights())

    def choose_action_max(self, observation, actions_set):  # 依据当前状态，计算有多少动作，并计算每个状态动作对的Q值，从而选择下一步动作(只选Q值最大的动作)

        input_data, _ = self.preprocess_CNNinput_last_obs1_actionSet_obs2([observation[0]], [actions_set], [observation[1]])
        input_data = self.patch_obs2arry(input_data)
        # ----------使用评估网络计算Q值-------
        # print('input_data', input_data)
        # input_data = normalization(input_data)  # [输入数据归一化到0-1]

        q_values = self.model_eval.call(input_data)
        # ---------选择Q值最大的动作---------
        q_values = np.ravel(q_values)  # 拉直为一维数组
        indexes = np.where(q_values == max(q_values))[0]  # where返回的是一个元组，如(array([0, 2], dtype=int64),)，将其中的数组剥离出来
        index = np.random.choice(indexes)  # 状态s下可能存在多个动作的Q值是相等的，从中任意选择一个
        best_action = actions_set[index]

        best_action = list(best_action)  # 要把元组形式转化为列表形式，以便后面操作
        return best_action

    def preprocess_CNNinput_obs1_action_obs2(self, Batch_obs1, Batch_action, Batch_obs2):
        """
        预处理exp第一个状态中的矩阵部分与动作部分
        """
        Batch_input_obs1_action_obs2 = []
        for b in range(self.batch_size):
            matrix_obs1 = self.transfer_state_to_matrix2(Batch_obs1[b], self.num_tasks)
            matrix_action = self.transfer_action_to_matrix2(Batch_action[b], self.num_tasks)
            matrix_obs1_action = np.vstack((matrix_obs1, matrix_action)).reshape((1, self.features_dim, self.num_tasks))
            matrix_obs1_action = matrix_obs1_action.swapaxes(0, 2)  # 将状态s和动作a作为网络的输入，形式为一个（6x活动数）的三维矩阵
            Batch_input_obs1_action_obs2.append([matrix_obs1_action, Batch_obs2[b]])
        # Batch_input_obs1_action_obs2 = np.array(Batch_input_obs1_action_obs2).astype('float32')
        return Batch_input_obs1_action_obs2

    def preprocess_CNNinput_last_obs1_actionSet_obs2(self, Batch_last_obs1, Actions_set_of_Batch_last_obs, Batch_last_obs2):
        """
        预处理exp最后一个状态中矩阵部分及其对应的所有可能的动作
        """
        Num_Actions = []  # 每个动作集合里的动作数量
        for x in Actions_set_of_Batch_last_obs:
            if x == [([], [])]:
                Num_Actions.append(0)
            else:
                Num_Actions.append(len(x))
        # ---构造目标网络的批量输入数据----
        Batch_input_last_obs1_actionSet_obs2 = []
        for b in range(len(Batch_last_obs1)):
            last_obs1 = Batch_last_obs1[b]
            actions_set = Actions_set_of_Batch_last_obs[b]
            for action in actions_set:
                # .......转化为矩阵形式........
                # print('action', action)
                matrix_last_obs1 = self.transfer_state_to_matrix2(last_obs1, self.num_tasks)
                matrix_action = self.transfer_action_to_matrix2(action, self.num_tasks)
                matrix_last_obs1_action = np.vstack((matrix_last_obs1, matrix_action)).reshape((1, self.features_dim, self.num_tasks))
                matrix_last_obs1_action = matrix_last_obs1_action.swapaxes(0, 2)  # 将当前状态动作对组合成目标网络的输入
                Batch_input_last_obs1_actionSet_obs2.append([matrix_last_obs1_action, Batch_last_obs2[b]])
        # Batch_input_last_obs1_actionSet_obs2 = np.array(Batch_input_last_obs1_actionSet_obs2).astype('float32')
        return Batch_input_last_obs1_actionSet_obs2, Num_Actions

    def transfer_action_to_matrix2(self, action, task_num):  # 把用活动编号形式表示动作转化为向量/矩阵形式表示
        # matrix_a = [[-1,-1,1,1,...-1],
        #             [-1,-1,0,2,...,-1]]

        activities = action[0]
        modes = action[1]
        row_act = np.array([-1 for _ in range(task_num)])  # 生成一个-1填充的向量
        row_mod = np.array([-1 for _ in range(task_num)])
        # print('action', action)
        if len(activities) != 0:
            for x in range(len(activities)):  # activities中的活动在矩阵中对应的位置为1，其他位置为0
                # print('x', x)
                task = activities[x]
                mod = modes[x]
                row_act[task - 1] = 1
                row_mod[task - 1] = mod
            matrix_a = np.asarray([row_act, row_mod])
        else:
            matrix_a = np.asarray([row_act, row_mod])
        return matrix_a

    def transfer_state_to_matrix2(self, obs_lists, task_num):  # 将已完成活动集合，正在进行的活动集合，各开始时间转化为16x6的矩阵
        # 状态observ = [[completed_tasks(列表)],[active_tasks(列表)],[M.value()]，[start_time.value()], [actCarbon.value()]]
        completed_tasks = obs_lists[0]  # 需转化
        active_tasks = obs_lists[1]  # 需转化
        selected_modes = obs_lists[2]  # 由字典的值组成的列表，无需处理
        start_time = obs_lists[3]  # 由字典的值组成的列表，需要标准化
        act_carbon = obs_lists[4]  # 由字典的值组成的列表，需要标准化
        row_C = [-1 for _ in range(task_num)]
        for task in completed_tasks:
            row_C[task - 1] = 1

        row_A = [-1 for _ in range(task_num)]
        for task in active_tasks:
            row_A[task - 1] = 1

        row_M = selected_modes

        # min_st = min(start_time)
        # max_st = max(start_time)
        min_st = -1
        max_st = self.max_makespan
        row_ST = [(x - min_st) / (max_st - min_st) for x in start_time]

        min_carbon = -1
        max_carbon = self.max_carbon
        row_CA = [(x - min_carbon) / (max_carbon - min_carbon) for x in act_carbon]
        # print('row_CA', row_CA)
        matrix_s0 = [row_C, row_A, row_M, row_ST, row_CA]
        matrix_s = np.array(matrix_s0, dtype='float32')

        return matrix_s

    def update_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon_end + (self.epsilon_start-self.epsilon_end)*math.exp(-(self.count_step*self.epsilon_decay_rate))  # 更新epsilon
        else:
            self.epsilon = self.epsilon_end
        self.epsilon_his.append(self.epsilon)

    def update_lr_cos(self, warm_episode, step, EPISODES, lr_max, lr_min):
        if step < warm_episode:
            lr = lr_max / warm_episode * step
        else:
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos((step - warm_episode) / EPISODES * np.pi))
        return lr

    def patch_obs2arry(self, Obs):
        # 原来的Obs格式为: 单个[array(), list]， 批量[[array(), list]]
        # 将Obs格式转化为: array
        if isinstance(Obs[0], list):
            Obs_array = []
            for b in range(len(Obs)):
                s1 = Obs[b][0]
                s2 = Obs[b][1]
                num_tasks = s1.shape[0]
                s2_add_zeros = [0 for _ in range(num_tasks - len(s2))]
                s2 = np.asarray(s2 + s2_add_zeros)
                s2 = np.reshape(s2, (num_tasks, 1, 1))
                s1and2 = np.concatenate((s1, s2), axis=1)
                Obs_array.append(s1and2)
            Obs_array = np.asarray(Obs_array)
        elif isinstance(Obs[0], np.ndarray):
            s1 = Obs[0]
            s2 = Obs[1]
            num_tasks = s1.shape[0]
            s2_add_zeros = [0 for _ in range(num_tasks - len(s2))]
            s2 = np.asarray(s2 + s2_add_zeros)
            s2 = np.reshape(s2, (num_tasks, 1, 1))
            s1and2 = np.concatenate((s1, s2), axis=1)
            Obs_array = s1and2
        else:
            print('ERROR: Agent-->def patch_obs2arry() 检查输入的状态数据的格式')
            Obs_array = None
        return Obs_array



