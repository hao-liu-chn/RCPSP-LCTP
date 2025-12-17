import matplotlib.pyplot as plt
import math
import mpl_toolkits.axisartist as axisartist
from matplotlib.ticker import MultipleLocator

def ladder_carbon_price(Quota, real_emission, gap, p_inc, init_carbon_price):
    if real_emission > Quota:
        temp = (real_emission-Quota)/Quota
        x = math.floor(temp/gap)
        carbon_price = init_carbon_price + x * p_inc
    else:
        temp = (Quota - real_emission) / Quota
        x = math.floor(temp / gap)
        carbon_price = -init_carbon_price - x * p_inc
    return carbon_price

if __name__ == '__main__':
    init_carbon_price = 0.05
    Quota = 500
    gap = 0.1
    p_inc = init_carbon_price/10
    R_E = list(range(0, 1000, 1))
    C_P = []
    X = []
    for real_emission in R_E:
        carbon_price = ladder_carbon_price(Quota, real_emission, gap, p_inc, init_carbon_price)
        C_P.append(carbon_price)
        X.append((real_emission - Quota) / Quota)

    # 设置全局字体和字号
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(4, 3))
    # 创建图形和坐标轴
    plt.plot(X, C_P, c='k')
    plt.xlabel('Tradable carbon emission / Q')
    plt.ylabel('Carbon price')
    plt.tight_layout()

    # 获取当前坐标轴对象
    ax = plt.gca()
    y_major_locator = MultipleLocator(0.025)  # 设置较小的刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)
    x_major_locator = MultipleLocator(0.3)  # 设置较小的刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    # 添加网格
    plt.grid(True)
    plt.show()


