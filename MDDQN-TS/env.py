import numpy as np
import itertools
import copy
from MoreStates import *
from ladder_carbon_price import *

class EnvCRCPSP:
    def __init__(self, task_num, succ, r_capacity, r_demand, S, Acts_space, Duration,
                 init_total_cost, theta, resource_price, energy_price, Re, Pe,
                 init_carbon_price, gap, p_inc, Energy_Consumption, Quota, D):  # 输入活动个数
        # 常量
        self.num = task_num
        self.successors = succ
        self.predecessors = self._get_predecessors()
        self.r_demand = r_demand  # 资源需求
        self.r_capacity = r_capacity  # 资源限量
        self.U = np.arange(1, self.num+1)  # 全部活动
        self.S_capacity = S  # 总可用面积
        self.Acts_Space = Acts_space  # 每个活动的资源需求
        self.Duration = Duration  # 每个活动在不同模式下的活动工期
        self.init_total_cost = init_total_cost  # 项目没开始时的总成本
        self.theta = theta
        self.resource_price = resource_price
        self.energy_price = energy_price
        self.Re = Re
        self.Pe = Pe
        self.init_carbon_price = init_carbon_price
        self.p_inc = p_inc
        self.gap = gap
        self.Energy_Consumption = Energy_Consumption
        self.Quota = Quota
        self.D = D

        # 变量
        self.C = [1]  # 已完成活动
        self.A = []  # 正在进行的活动
        self.M = {j: -1 for j in range(1, self.num+1)}  # 速度模式
        self.M[1] = 0
        self.startTime = {j: -1 for j in range(1, self.num+1)}  # 记录每个活动的开始时间
        self.finishTime = {j: -1 for j in range(1, self.num+1)}  # 记录每个活动的结束时间
        self.startTime[1] = 0
        self.finishTime[1] = 0
        self.actCarbon = {j: -1 for j in range(1, self.num + 1)}
        self.actCarbon[1] = 0
        self.r_remain = self.r_capacity  # 初始状态的剩余资源
        self.S_remain = self.S_capacity  # 初始状态下的剩余工作面
        self.Time = 0  # 记录项目执行过程中的当前时间
        self.oldTime = 0  # 记录上时刻的时间
        self.total_cost = None
        self.old_total_cost = self.init_total_cost

    def reset(self):
        self.C = [1]
        self.A = []
        self.M = {j: -1 for j in range(1, self.num + 1)}  # 速度模式
        self.M[1] = 0
        self.startTime = {j: -1 for j in range(1, self.num + 1)}  # 记录每个活动的开始时间
        self.finishTime = {j: -1 for j in range(1, self.num + 1)}  # 记录每个活动的结束时间
        self.startTime[1] = 0
        self.finishTime[1] = 0
        self.actCarbon = {j: -1 for j in range(1, self.num+1)}
        self.actCarbon[1] = 0
        self.r_remain = self.r_capacity  # 初始状态的剩余资源
        self.S_remain = self.S_capacity  # 初始状态下的剩余工作面
        self.Time = 0  # 记录项目执行过程中的当前时间
        self.oldTime = 0  # 记录上时刻的时间
        self.total_cost = None
        self.old_total_cost = self.init_total_cost
        UNC = unscheduled_NC(self.num, self.C, self.A, self.successors)
        URS = unscheduled_RS(self.num, self.C, self.A, self.r_capacity, self.r_demand, single=True)
        URC = unscheduled_RC(self.num, self.C, self.A, self.r_capacity, self.r_demand, single=True)
        UAD = unscheduled_AD(self.num, self.C, self.A, self.Duration)
        USS = unscheduled_Space_strength(self.num, self.C, self.A, self.Acts_Space, self.S_capacity)
        obs_matrix = [self.C,
                      self.A,
                      list(self.M.values()),
                      list(self.startTime.values()),
                      list(self.actCarbon.values())]
        # obs_indicators = UNC
        obs_indicators = UNC + URS + URC + UAD + USS
        observation = [obs_matrix, obs_indicators]   # 初始状态
        return observation

    def step(self, action_):
        new_observation = self._update_obs(action_)
        reward = self._get_reward()
        done = self._get_down()
        return new_observation, reward, done

    def _update_obs(self, action):  # 获取采取该动作后得到的新状态  # 要考虑几个问题：(某个状态下，动作集合可能是空集，即紧前活动都已完成的活动的均在集合A中)
        # ----更新新执行的活动的开始、结束时间、资源剩余量、速度模式----
        activities = action[0]  # list
        modes = action[1]  # list
        for x in range(len(activities)):
            task = activities[x]
            mod = modes[x]
            self.startTime[task] = self.Time  # 更新开始时间
            self.finishTime[task] = self.Time + self.Duration[task][mod]  # 更新结束时间
            self.M[task] = mod  # 更新模式
            self.r_remain = list(np.asarray(self.r_remain) - np.asarray(self.r_demand[task]))
            self.S_remain -= self.Acts_Space[task]

        self.oldTime = self.Time
        temp = self.A + activities  # 上一代正在进行的活动集合+本阶段新安排的活动集合
        self.Time = np.amin(np.asarray(list(self.finishTime.values()))[np.asarray(temp)-1])
        # ----更新已完成活动集合C及剩余资源r_remain----
        finished_tasks = np.asarray(temp)[np.asarray(list(self.finishTime.values()))[np.asarray(temp)-1] == self.Time]  # 在当前时刻结束的活动
        self.C = self.C + list(finished_tasks)
        for fin_task in finished_tasks:  # 新增已完成活动释放出的资源纳入剩余闲置资源中,释放出的空间纳入闲置空间
            self.r_remain = list(np.asarray(self.r_remain) + np.asarray(self.r_demand[fin_task]))
            self.S_remain += self.Acts_Space[fin_task]

        # ----更新正在进行的活动集合A----
        self.A = list(set(temp)-set(finished_tasks))

        # ----判断当前时刻是否为决策点----
        temp_eligible = self._get_eligible(self.A, self.C)
        decision_point = False
        if activities == [self.num]:
            decision_point = True
        elif len(temp_eligible) != 0:
            for tem_eli in temp_eligible:
                if self.r_demand[tem_eli][0] <= self.r_remain[0] \
                        and self.r_demand[tem_eli][1] <= self.r_remain[1] \
                        and self.r_demand[tem_eli][2] <= self.r_remain[2] \
                        and self.r_demand[tem_eli][3] <= self.r_remain[3] \
                        and self.Acts_Space[tem_eli] <= self.S_remain:
                    decision_point = True  # 此刻如果闲置的资源能够支持可行活动集合中任何一个活动开始时的资源需求，则当前时刻为决策点
                    break
        else:
            decision_point = False

        while not decision_point:
            # ----更新当前时间----
            self.Time = np.amin(np.asarray(list(self.finishTime.values()))[np.asarray(self.A) - 1])
            # ----更新已完成活动集合C及剩余资源r_remain----
            finished_tasks = np.asarray(self.A)[np.asarray(list(self.finishTime.values()))[np.asarray(self.A)-1] == self.Time]
            self.C = self.C + list(finished_tasks)
            for fin_task in finished_tasks:  # 新增已完成活动释放出的资源纳入剩余闲置资源中,释放的空间也纳入闲置空间
                self.r_remain = list(np.asarray(self.r_remain) + np.asarray(self.r_demand[fin_task]))
                self.S_remain += self.Acts_Space[fin_task]
            # ----更新正在进行的活动集合A----
            self.A = list(set(self.A) - set(finished_tasks))

            temp_eligible = self._get_eligible(self.A, self.C)
            if len(temp_eligible) != 0:  # 如果此刻可行活动集合不是空集，则判断可行活动集合中有没有活动满足资源约束可以执行的
                for eli_task in temp_eligible:  # 依此判断可行活动集合中的活动，如果有活动满足资源约束，则当前时刻是一个决策点
                    if self.r_demand[eli_task][0] <= self.r_remain[0] \
                            and self.r_demand[eli_task][1] <= self.r_remain[1] \
                            and self.r_demand[eli_task][2] <= self.r_remain[2] \
                            and self.r_demand[eli_task][3] <= self.r_remain[3] \
                            and self.Acts_Space[eli_task] <= self.S_remain:
                        decision_point = True  # 此刻如果闲置的资源能够支持可行活动集合中任何一个活动开始时的资源需求，则当前时刻为决策点
                        break
                    else:  # 当前剩余的资源不足以支持可行活动集合中任一活动的执行
                        decision_point = False
            else:  # 此时可行活动为空集（即所有的满足优先关系的活动都正在执行中）
                decision_point = False
        for j in range(1, self.num):
            speed_mode = self.M[j]
            if speed_mode != -1:
                self.actCarbon[j] = self.Energy_Consumption[j][speed_mode] * self.theta
        new_obs_matrix = [self.C,
                          self.A,
                          list(self.M.values()),
                          list(self.startTime.values()),
                          list(self.actCarbon.values())]
        new_UNC = unscheduled_NC(self.num, self.C, self.A, self.successors)
        new_URS = unscheduled_RS(self.num, self.C, self.A, self.r_capacity, self.r_demand, single=True)
        new_URC = unscheduled_RC(self.num, self.C, self.A, self.r_capacity, self.r_demand, single=True)
        new_UAD = unscheduled_AD(self.num, self.C, self.A, self.Duration)
        new_USS = unscheduled_Space_strength(self.num, self.C, self.A, self.Acts_Space, self.S_capacity)
        # new_obs_indicators = new_UNC
        new_obs_indicators = new_UNC + new_URS + new_URC + new_UAD + new_USS
        new_observation = [new_obs_matrix, new_obs_indicators]
        return new_observation

    def _get_reward(self):  # 获取当前即时奖励
        self.total_cost, _, _, _, _ = self.get_total_cost()
        # self.total_cost = self.get_total_cost2()
        reward = -(self.total_cost - self.old_total_cost) / 100
        self.old_total_cost = self.total_cost

        return reward

    def get_actions_set(self, observation):  # 计算在当前状态下有多少个可选动作（要考虑每个活动有多个模式）
        temp_act_set = []  # 可能的活动集合 [[2],[2, 3],...[2, 3, 4]]  只有活动，没有速度模式
        act_set = []  # 活动集合 在temp_act_set基础上去除无效率的活动组合
        actions_set = []  # 储存可行的动作 [([act2, act3],[mod1, mod0]),...,(),())]
        A = observation[0][1]  # 正在进行的活动集合
        C = observation[0][0]  # 已完成活动集合
        # 根据当前状态里的正在进行的活动来计算此状态下的资源剩余量和工作面剩余量
        R_remain = copy.deepcopy(self.r_capacity)
        S_remain = copy.deepcopy(self.S_capacity)
        for active_task in A:  # 正在进行的活动集合
            R_remain[0] = R_remain[0] - self.r_demand[active_task][0]
            R_remain[1] = R_remain[1] - self.r_demand[active_task][1]
            R_remain[2] = R_remain[2] - self.r_demand[active_task][2]
            R_remain[3] = R_remain[3] - self.r_demand[active_task][3]
            S_remain = S_remain - self.Acts_Space[active_task]

        eligible = self._get_eligible(A, C)  # 获得可行活动集合  （如果eligible是空集呢？eligible是空集的唯一情况就是所有活动都安排完了，observation是项目结束的最终状态）
        if len(eligible) != 0:
            # ----列出可行活动集合的全部子集----
            all_subsets = []  # 储存可行活动集合的全部子集
            for n in range(1, len(eligible)+1):
                for subset in itertools.combinations(eligible, n):
                    all_subsets.append(list(subset))

            # ----删除不符合资源约束和空间约束的子集----
            for i in range(len(all_subsets)):  # 检查每个子集
                r_need = np.zeros(4, int)
                s_need = 0
                for j in range(len(all_subsets[i])):  # 计算该子集需要占用多少资源
                    task = all_subsets[i][j]
                    r_need += self.r_demand[task]
                    s_need += self.Acts_Space[task]

                if r_need[0] <= R_remain[0] \
                        and r_need[1] <= R_remain[1] \
                        and r_need[2] <= R_remain[2] \
                        and r_need[3] <= R_remain[3] \
                        and s_need <= S_remain:
                    temp_act_set.append(all_subsets[i])  # 符合条件的子集纳入活动集合

            # ----删除对剩余资源利用不充分的子集----
            # （例如如果可以同时执行活动2,3，那么没必要只执行活动2或活动3）
            for act in temp_act_set:
                remain_eligible_tasks = list(set(eligible)-set(act))
                if len(remain_eligible_tasks) == 0:
                    is_efficient = True
                else:
                    temp_r_remain = copy.deepcopy(R_remain)
                    temp_s_remain = copy.deepcopy(S_remain)
                    for task in act:  # 执行了这个动作后，还能剩下多少资源和工作面
                        temp_r_remain = list(np.asarray(temp_r_remain)-np.asarray(self.r_demand[task]))
                        temp_s_remain -= self.Acts_Space[task]

                    is_efficient = True
                    for re_task in remain_eligible_tasks:  # 检查剩余的可行活动还能不能被执行
                        if self.r_demand[re_task][0] <= temp_r_remain[0] \
                                and self.r_demand[re_task][1] <= temp_r_remain[1] \
                                and self.r_demand[re_task][2] <= temp_r_remain[2] \
                                and self.r_demand[re_task][3] <= temp_r_remain[3] \
                                and self.Acts_Space[re_task] <= temp_s_remain:
                            is_efficient = False  # 只要有一个剩余的可行活动可以被执行，则该子集不是有效的动作
                            break
                if is_efficient:
                    act_set.append(act)
            # ----给每个活动组合配备速度模式-----
            Modes = [0, 1, 2]
            num_modes = len(Modes)
            for acts in act_set:
                # 获得acts的模式组合
                patterns = [Modes for _ in range(len(acts))]  # 生成每个元素对应的三种模式
                mode_combinations = [[patterns[j][i // (num_modes ** j) % num_modes] for j in range(len(acts))]
                                     for i in range(num_modes ** len(acts))]  # 生成全部模式组合
                for comb in mode_combinations:
                    actions_set.append((acts, comb))
        else:  # 可行活动集合为空集
            actions_set = [([], [])]
        # print('actions_set', actions_set)
        return actions_set

    def _get_predecessors(self):  # 获得紧前关系
        pre = {n + 1: [] for n in range(self.num)}
        for i in range(self.num - 1):
            temp = self.successors[i + 1]
            for j in range(len(temp)):
                pre[temp[j]].append(i + 1)
        return pre

    def _get_down(self):
        if self.finishTime[self.num] != -1:
            done = True
        else:
            done = False
        return done

    # def check_obs(self, observation, qtable):  # 检查该状态是不是已经存在于q_table中
    #     if observation not in qtable:  # 如果状态不在q_table中，则给其状态-动作对的Q值赋值为0
    #         actions_set = self.get_actions_set(observation)
    #         temp_dict = {}
    #         for act in actions_set:  # 当最后一个活动被安排后，这个状态对应的actions_set是空集
    #             temp_dict[tuple(act)] = 0
    #         qtable[observation] = temp_dict
    #     return qtable

    def _get_eligible(self, active_tasks, completed_tasks):  # 获得合格活动集合（仅满足优先关系即可）
        unschedule = list(set(self.U) - set(active_tasks) - set(completed_tasks))
        eligible = []  # 可行活动集合
        for task in unschedule:
            temp_pre = self.predecessors[task]  # 未调度活动的紧前活动
            if set(temp_pre).issubset(completed_tasks):  # 未调度活动的紧前活动是否均已完成
                eligible.append(task)
        return eligible

    def get_total_cost(self):  # 计算已调度活动的总成本
        AplusC = self.A + self.C  # 已安排的活动
        res_cost = 0  # 资源使用成本
        energy_cost = 0  # 能源成本
        all_carbon_emission = 0  # 已安排活动总碳排放

        for task in AplusC:
            mod = self.M[task]
            res_cost += sum(np.asarray(self.resource_price) * np.asarray(self.r_demand[task])) * self.Duration[task][mod]
            energy_cost += self.energy_price * self.Energy_Consumption[task][mod]
            all_carbon_emission += self.Energy_Consumption[task][mod] * self.theta

        if self.Time <= self.D:
            ms_reward = (self.D - self.Time) * self.Re
            ms_punish = 0
        else:
            ms_reward = 0
            ms_punish = (self.Time - self.D) * self.Pe
        carbon_price = ladder_carbon_price(self.Quota, all_carbon_emission, self.gap, self.p_inc, self.init_carbon_price)
        # print('carbon_price', carbon_price)
        carbon_trading = abs(self.Quota - all_carbon_emission) * carbon_price
        total_cost = res_cost + energy_cost - ms_reward + ms_punish + carbon_trading

        return total_cost, res_cost, energy_cost, carbon_trading, all_carbon_emission

    def get_total_cost2(self):  # 计算已调度活动的总成本
        AplusC = self.A + self.C  # 已安排的活动
        Unscheduled_tasks = [j for j in range(1, self.num+1) if j not in AplusC]  # 未安排活动
        res_cost = 0  # 资源使用成本
        energy_cost = 0  # 能源成本
        all_carbon_emission = 0  # 已安排活动总碳排放

        for task in AplusC:
            mod = self.M[task]
            res_cost += sum(np.asarray(self.resource_price) * np.asarray(self.r_demand[task])) * self.Duration[task][mod]
            energy_cost += self.energy_price * self.Energy_Consumption[task][mod]
            all_carbon_emission += self.Energy_Consumption[task][mod] * self.theta

        for task in Unscheduled_tasks:
            all_carbon_emission += self.Energy_Consumption[task][0] * self.theta  # 未调度活动碳排放按最小算（最慢速度）

        if self.Time <= self.D:
            ms_reward = (self.D - self.Time) * self.Re
            ms_punish = 0
        else:
            ms_reward = 0
            ms_punish = (self.Time - self.D) * self.Pe
        carbon_price = ladder_carbon_price(self.Quota, all_carbon_emission, self.gap, self.p_inc, self.init_carbon_price)
        carbon_trading = abs(self.Quota - all_carbon_emission) * carbon_price
        total_cost = res_cost + energy_cost - ms_reward + ms_punish + carbon_trading

        return total_cost

    def get_total_cost3(self):  # 未调度活动按最低排放计算，算出总排放后，计算碳价格，计算碳交易时用已安排活动的碳排放+未安排的最低碳排放
        AplusC = self.A + self.C  # 已安排的活动
        Unscheduled_tasks = [j for j in range(1, self.num+1) if j not in AplusC]  # 未安排活动
        res_cost = 0  # 资源使用成本
        energy_cost = 0  # 能源成本
        all_carbon_emission = 0  # 已安排活动总碳排放

        for task in AplusC:
            mod = self.M[task]
            res_cost += sum(np.asarray(self.resource_price) * np.asarray(self.r_demand[task])) * self.Duration[task][mod]
            energy_cost += self.energy_price * self.Energy_Consumption[task][mod]
            all_carbon_emission += self.Energy_Consumption[task][mod] * self.theta

        un_carbon_emission = 0
        for task in Unscheduled_tasks:
            un_carbon_emission += self.Energy_Consumption[task][0] * self.theta  # 未调度活动碳排放按最小算（最慢速度）
        temp_emission = all_carbon_emission + un_carbon_emission

        if self.Time <= self.D:
            ms_reward = (self.D - self.Time) * self.Re
            ms_punish = 0
        else:
            ms_reward = 0
            ms_punish = (self.Time - self.D) * self.Pe
        carbon_price = ladder_carbon_price(self.Quota, temp_emission, self.gap, self.p_inc, self.init_carbon_price)
        carbon_trading = abs(self.Quota - temp_emission) * carbon_price
        total_cost = res_cost + energy_cost - ms_reward + ms_punish + carbon_trading

        return total_cost

    def get_total_cost4(self):  # 按初始碳价格计算，计算碳交易时仅用已安排活动的碳排放
        AplusC = self.A + self.C  # 已安排的活动
        Unscheduled_tasks = [j for j in range(1, self.num+1) if j not in AplusC]  # 未安排活动
        res_cost = 0  # 资源使用成本
        energy_cost = 0  # 能源成本
        all_carbon_emission = 0  # 已安排活动总碳排放

        for task in AplusC:
            mod = self.M[task]
            res_cost += sum(np.asarray(self.resource_price) * np.asarray(self.r_demand[task])) * self.Duration[task][mod]
            energy_cost += self.energy_price * self.Energy_Consumption[task][mod]
            all_carbon_emission += self.Energy_Consumption[task][mod] * self.theta

        if self.Time <= self.D:
            ms_reward = (self.D - self.Time) * self.Re
            ms_punish = 0
        else:
            ms_reward = 0
            ms_punish = (self.Time - self.D) * self.Pe
        carbon_price = self.init_carbon_price
        carbon_trading = abs(self.Quota - all_carbon_emission) * carbon_price
        total_cost = res_cost + energy_cost - ms_reward + ms_punish + carbon_trading

        return total_cost