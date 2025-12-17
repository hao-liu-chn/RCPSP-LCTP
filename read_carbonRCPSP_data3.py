# 读取算例数据


def read_carbon_rcpsp_file(file_path_):
    successors_ = {}
    resource_demand_ = {}
    wight_speed_ = {}
    quality_basic_ = {}
    wight_quality_ = {}
    work_load = {}  # 没用 可删除
    duration_basic = {}

    with open(file_path_) as f:
        content = f.readlines()
        J_ = int(content[1][5:-1])
        num_r_ = int(content[2][22:-1])

        for line in range(len(content)):
            # 读取优先关系(紧后活动)
            start_a = 6
            end_a = start_a + J_ - 1
            if start_a <= line <= end_a:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                jobnr = line_content[0]
                succ_tasks = line_content[2:]
                successors_[jobnr] = succ_tasks

            # 读取资源需求
            start_r = end_a + 3
            end_r = start_r + J_ - 1
            if start_r <= line <= end_r:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                jobnr = line_content[0]
                demand = line_content[1:]
                resource_demand_[jobnr] = demand

            # 读取资源可用量
            position_R = end_r + 3
            if line == position_R:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                r_availability = line_content

            # 读取速度权重
            start_speed = position_R + 3
            end_speed = start_speed + J_ - 1
            if start_speed <= line <= end_speed:
                line_content = content[line].split()
                line_content = list(map(float, line_content))
                jobnr = line_content[0]
                w_s = line_content[1:]
                wight_speed_[int(jobnr)] = w_s

            # 读取活动质量（基础）
            position_quality = end_speed + 2
            if line == position_quality:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                for x in range(1, J_ + 1):
                    quality_basic_[x] = line_content[x - 1]

            # 读取活动质量权重
            position_wight_quality = position_quality + 2
            if line == position_wight_quality:
                line_content = content[line].split()
                line_content = list(map(float, line_content))
                for x in range(1, J_ + 1):
                    wight_quality_[x] = line_content[x - 1]

            # 读取资源占用空间
            position_resource_space = position_wight_quality + 3
            if line == position_resource_space:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                resource_space = line_content

            # 读取项目可用空间量
            position_space_av = position_resource_space + 2
            if line == position_space_av:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                space_availability = line_content[0]

            # 读取速度（基础）
            position_speed = position_space_av + 3
            if line == position_speed:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                resource_speed = line_content

            # 读取WORK LOAD
            position_work_load = position_speed + 2
            if line == position_work_load:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                for x in range(1, J_ + 1):
                    work_load[x] = line_content[x - 1]

            # 读取DELTA
            position_delta = position_work_load + 2
            if line == position_delta:
                line_content = content[line].split()
                line_content = list(map(float, line_content))
                delta = line_content

            # 读取基础活动工期
            position_duration = position_delta + 2
            if line == position_duration:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                for j in range(1, J_+1):
                    duration_basic[j] = line_content[j-1]

            # 读取活动工期
            position_c_ms = position_duration + 2
            if line == position_c_ms:
                line_content = content[line].split()
                line_content = list(map(int, line_content))
                D = line_content[0]

    return J_, num_r_, successors_, resource_demand_, r_availability, wight_speed_, quality_basic_, wight_quality_, \
           resource_space, space_availability, resource_speed, work_load, delta, duration_basic, D


# # ========【测试】============
# file_path = 'E:\\workspace\\MyCode\\Python\\单目标-EDA0227-RL-局部搜索\\instances\\test\\j1010_1.txt'
# J_, num_r_, successors_, resource_demand_, r_availability, wight_speed_, quality_basic_, \
# wight_quality_, resource_space, space_availability, resource_speed, work_load, delta, duration_basic \
#     = read_carbon_rcpsp_file(file_path)
# print('successors_=', successors_)
# print('resource_demand_=', resource_demand_)
# print('r_availability=', r_availability)
# print('wight_speed_=', wight_speed_)
# print('quality_basic_=', quality_basic_)
# print('wight_quality_=', wight_quality_)
# print('resource_space=', resource_space)
# print('space_availability=', space_availability)
# print('resource_speed=', resource_speed)
# print('work_load=', work_load)
# print('delta=', delta)
# print('duration_basic=', duration_basic)
