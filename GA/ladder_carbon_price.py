import matplotlib.pyplot as plt
import math

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
    init_carbon_price = 0.1
    Quota = 1000
    gap = 0.1
    p_inc = init_carbon_price/10
    R_E = list(range(500, 1500, 1))
    C_P = []
    X = []
    for real_emission in R_E:
        carbon_price = ladder_carbon_price(Quota, real_emission, gap, p_inc, init_carbon_price)
        C_P.append(carbon_price)
        X.append((real_emission - Quota) / Quota / gap)
    plt.plot(R_E, C_P)
    plt.show()

