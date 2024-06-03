import copy
import os
import random

Max_iter = [100, 200]
Tabu_tenure = [3, 5, 7]
Tabu_size = [5, 10, 15]
Num_neigh = [3, 5, 10]
num = 0
for max_iterations in Max_iter:
    for tabu_tenure in Tabu_tenure:
        for tabu_size in Tabu_size:
            for num_neighbors in Num_neigh:
                Inst_Set = ['all2']
                for inst_set in Inst_Set:
                    # [local file path]
                    path_inst_set = "E:/workspace/Project Scheduling/CarbonRCSPS Instances/Orth/" + inst_set
                    Inst_List = os.listdir(path_inst_set)
                    with open('tabu_exp.py', 'r', encoding='utf-8') as f:
                        code_str = f.read()
                    num += 1
                    # 修改参数
                    code_str = code_str.replace('max_iterations = 100', 'max_iterations = '+str(max_iterations))
                    code_str = code_str.replace("tabu_tenure = 3", "tabu_tenure = " + str(tabu_tenure))
                    code_str = code_str.replace('tabu_size = 20', 'tabu_size = ' + str(tabu_size))
                    code_str = code_str.replace("num_neighbors1 = 5", "num_neighbors1 = " + str(num_neighbors))
                    code_str = code_str.replace('num_neighbors2 = 5', 'num_neighbors2 = ' + str(num_neighbors))
                    # 另存为新的文件
                    with open('tabu_exp_'+inst_set+'_' + str(num) + '.py', 'w') as f:
                        f.write(code_str)

                    print('成功')


