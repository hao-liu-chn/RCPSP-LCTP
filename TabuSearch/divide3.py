import copy
import os
import random

Inst_Set = ['carbon-j10-sample', 'carbon-j20-sample', 'carbon-j30-sample', 'carbon-j60-sample', 'carbon-j90-sample', 'carbon-j120-sample']

for inst_set in Inst_Set:
    # [local file path]
    path_inst_set = "E:/workspace/Project Scheduling/CarbonRCSPS Instances/sample/" + inst_set
    Inst_List = os.listdir(path_inst_set)

    num_inst = 100
    interval = 10
    s_inst = []
    x = 0
    while x < num_inst:
        s_inst.append(x)
        x += interval
    e_inst = copy.deepcopy(s_inst[1:])
    e_inst.append(num_inst)

    num = 0
    for i in range(len(s_inst)):
        num += 1
        start = s_inst[i]
        end = e_inst[i]
        # 读取原始代码文件
        with open('tabu_exp2_10000.py', 'r', encoding='utf-8') as f:
            code_str = f.read()

        # 修改参数
        code_str = code_str.replace('for u in range(0, 1)', 'for u in range('+str(start)+', '+str(end)+')')
        code_str = code_str.replace("Inst_Set = 'carbon-j10-sample'", "Inst_Set = '" + inst_set + "'")
        # 另存为新的文件
        with open('tabu_exp2_10000_'+inst_set+'_' + str(num) + '.py', 'w') as f:
            f.write(code_str)

    print('成功')


