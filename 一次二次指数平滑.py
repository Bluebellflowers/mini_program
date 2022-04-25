real_capacity = [47118, 51025, 55454, 60152, 68068]
# 取第一年的预测值等于实际值
pre_capacity = [47118]
from copy import deepcopy


def single_exponential_smoothing(real_capacity, pre_capacity, a):
    real = deepcopy(real_capacity)
    pre = deepcopy(pre_capacity)
    for index in range(len(real)):
        next_pre = a * real[index] + (1 - a) * pre[index]
        pre.append(round(next_pre, 2))
    return pre


pre_0_1 = single_exponential_smoothing(real_capacity, pre_capacity, a=0.1)
pre_0_9 = single_exponential_smoothing(real_capacity, pre_capacity, a=0.9)
import matplotlib.pyplot as plt

plt.figure()
# 绘制alpha等于0.1时
plt.plot([i for i in range(2014, 2020)], pre_0_1, label='alpha=0.1')
# 绘制alpha=0.9时
plt.plot([i for i in range(2014, 2020)], pre_0_9, label='alpha=0.9')
# 绘制实际值
plt.plot([i for i in range(2014, 2019)], real_capacity, label='real')
plt.xlabel("year")
plt.ylabel("capacity")
plt.legend()
plt.show()
