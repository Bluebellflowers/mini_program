import pandas as pd
import random
from math import sqrt

or_data = pd.read_excel('./or_data.xlsx')

# 安全半径
sr = sqrt((or_data['纬度'].max() - or_data['纬度'].min()) ** 2 + (or_data['经度'].max() - or_data['经度'].min()) ** 2)

N = 5  # 车辆数目
KN = 100  # 迭代限定值
kn = 0
i = 0
ac = []
loop=0
# while i < N:
#     # 随机生成车辆i的配送中心坐标
#     loop+=1
#     # temp=[random.random()*(or_data['纬度'].max()-or_data['纬度'].min()),random.random()*(or_data['经度'].max()-or_data['经度'].min())]
#     if i == 0:
#         ac.append([random.uniform(or_data['纬度'].min(), or_data['纬度'].max()),
#                    random.uniform(or_data['经度'].min(), or_data['经度'].max())])
#         i += 1
#     else:
#         temp = [0.5*random.uniform(or_data['纬度'].min(), 1.5*or_data['纬度'].max()),
#                 0.5*random.uniform(or_data['经度'].min(), 1.5*or_data['经度'].max())]
#         ac.append(temp)
#         i += 1
#         if i > 1:  # 若有两个中心以上，则需要进行安全半径判断
#             distance = []
#             for center in ac[:i - 1]:
#                 distance.append(sqrt((ac[-1][0] - center[0]) ** 2 + (ac[-1][1] - center[1]) ** 2))
#             if [ii for ii in distance if ii >= sr]:
#                 print(temp)
#                 if i == N:  # 满足条件，结束循环
#                     print(ac)
#                     break
#             else:
#                 ac.pop()
#                 i -= 1


while i < N:
    # 随机生成车辆i的配送中心坐标
    loop+=1
    # temp=[random.random()*(or_data['纬度'].max()-or_data['纬度'].min()),random.random()*(or_data['经度'].max()-or_data['经度'].min())]
    if i == 0:
        ac.append([random.uniform(or_data['纬度'].min(), or_data['纬度'].max()),random.uniform(or_data['经度'].min(), or_data['经度'].max())])
        i += 1
    else:
        temp=[random.random()*(or_data['纬度'].max()-or_data['纬度'].min()),random.random()*(or_data['经度'].max()-or_data['经度'].min())]
        # temp = [0.5*random.uniform(or_data['纬度'].min(), 1.5*or_data['纬度'].max()),
        #         0.5*random.uniform(or_data['经度'].min(), 1.5*or_data['经度'].max())]
        ac.append(temp)
        i += 1
        if i > 1:  # 若有两个中心以上，则需要进行安全半径判断
            distance = []
            for center in ac[:i - 1]:
                distance.append(sqrt((ac[-1][0] - center[0]) ** 2 + (ac[-1][1] - center[1]) ** 2))
            if [ii for ii in distance if ii >= sr]:
                print(temp)
                if i == N:  # 满足条件，结束循环
                    print(ac)
                    print(loop)
                    break
            else:
                ac.pop()
                i -= 1
