import math
from matplotlib import pyplot as plt
from copy import deepcopy

arrive_rate = [0.59, 0.94, 0.97, 0.59, 0.24, 0.5, 0.89, 0.16, 0.78, 0.98, 0.42, 0.03, 0.35, 0.28, 0.29, 0.55,
               0.81, 0.78, 0.87, 0.1]
avg1 = 10
avg2 = 3
mu = 0.2


def time(x, lamada):  # 负指数分布
    return math.log(1 - x) / (-lamada)


time_gap = []
arrival_time = []
t = 8
for i in arrive_rate:
    if t <= 9:
        temp = time(i, avg1)
        t = t + temp
        arrival_time.append(t)
        time_gap.append(temp)
    else:
        temp = time(i, avg2)
        t = t + temp
        arrival_time.append(t)
        time_gap.append(temp)
temp = []
for var in arrival_time:
    if var <= 12:
        temp.append(var)
arrival_time = temp

i = 0
j = 0
leave_time = arrival_time.copy()
que_length = 0
que_length_list = [0]
leave_time[j] = arrival_time[i] + mu
while True:
    while leave_time[j] > arrival_time[i + 1]:
        que_length = que_length + 1
        i = i + 1
        que_length_list.append(que_length)
        if i == len(arrival_time) - 1:
            que_length_list.append(que_length)
            while (j + 1) < i:
                que_length = que_length - 1
                j = j + 1
                leave_time[j] = leave_time[j - 1] + mu
                que_length_list.append(que_length)
            break
    if j == i:
        que_length_list.append(0)
        i = i + 1
        j = j + 1
        leave_time[j] = arrival_time[i] + mu
        que_length = 0
        que_length_list.append(que_length)
    else:
        que_length = que_length - 1
        que_length_list.append(que_length)
        j = j + 1
        leave_time[j] = leave_time[j - 1] + mu
    if j == len(arrival_time) - 1:
        break
print('车辆的到达时间为：')
print(arrival_time)
print('车辆的离开时间为：')
print(leave_time)
print('车辆的最晚离开时间为：{}'.format(leave_time[-1]))
plt.plot(arrival_time, range(len(arrival_time)), label='arrival time')
plt.plot(leave_time, range(len(arrival_time)), label='leave time')
plt.xlabel('time')
plt.ylabel('vehicle number')
plt.legend()
plt.show()

wait_time = []
for i in range(len(arrival_time)):
    wait_time.append(leave_time[i] - arrival_time[i])
print('车辆的排队时间为：')
print(wait_time)
print('车辆的最大排队时间为：{}'.format(max(wait_time)))
plt.plot(range(len(wait_time)),wait_time,label='wait time')
plt.xlabel('time')
plt.ylabel('wait time')
plt.legend()
plt.show()
print('队列长度为：')
print(que_length_list)
print('最大队列长度为：{}'.format(max(que_length_list)))
print('平均排队时间为：{}'.format(sum(wait_time)/len(wait_time)))
plt.plot(range(len(que_length_list)),que_length_list,label='que length')
plt.xlabel('time')
plt.ylabel('que length')
plt.legend()
plt.show()
