# -*- coding: utf-8 -*-
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


def calDistance(CityCoordinates):
    '''
    计算城市间距离
    输入：CityCoordinates-城市坐标；
    输出：城市间距离矩阵-dis_matrix
    '''
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)
    return dis_matrix


def assign_distribution_center(dis_matrix, DC, C):
    '''
    Parameters:
    dis_matrix : 距离矩阵
    DC : 配送中心个数
    C : 客户点个数
    Returns:
    d-存储分配给配送中心的客户点，list最后嵌套list
    '''
    d = [[] for i in range(DC)]  # 存储分配的列表
    for i in range(DC, DC + C):
        d_i = [dis_matrix.loc[i, j] for j in range(DC)]  # 取出当前客户分别距离配送中心的距离
        min_dis_index = d_i.index(min(d_i))  # 取出最近的配送中心

        d[min_dis_index].append(i)  # 将客户点分配给配送中心

    return d


def greedy(CityCoordinates, dis_matrix, DC, certer_number):
    '''
    贪婪策略构造初始解
    输入：CityCoordinates-节点坐标,dis_matrix-距离矩阵
    输出：初始解-line
    '''
    # 修改dis_matrix以适应求解需要
    dis_matrix = dis_matrix.iloc[[certer_number] + CityCoordinates, [certer_number] + CityCoordinates].astype(
        'float64')  # 只取当前需要的配送中心和客户点
    for i in CityCoordinates: dis_matrix.loc[i, i] = math.pow(10, 10)
    dis_matrix.loc[:, certer_number] = math.pow(10, 10)  # 配送中心不在编码内
    line = []  # 初始化
    now_cus = random.sample(CityCoordinates, 1)[0]  # 随机生成出发点
    line.append(now_cus)  # 添加当前城市到路径
    dis_matrix.loc[:, now_cus] = math.pow(10, 10)  # 更新距离矩阵，已经过客户点不再被取出
    for i in range(1, len(CityCoordinates)):
        next_cus = dis_matrix.loc[now_cus, :].idxmin()  # 距离最近的城市
        line.append(next_cus)  # 添加进路径
        dis_matrix.loc[:, next_cus] = math.pow(10, 10)  # 更新距离矩阵
        now_cus = next_cus  # 更新当前城市
    return line


def calFitness(birdPop, certer_number, Demand, dis_matrix, CAPACITY, DISTABCE, C0, C1):
    '''
    贪婪策略分配车辆（解码），计算路径距离（评价函数）
    输入：birdPop-路径，Demand-客户需求,dis_matrix-城市间距离矩阵，CAPACITY-车辆最大载重,DISTABCE-车辆最大行驶距离,C0-车辆启动成本,C1-车辆单位距离行驶成本；
    输出：birdPop_car-分车后路径,fits-适应度
    '''
    birdPop_car, fits = [], []  # 初始化
    for j in range(len(birdPop)):
        bird = birdPop[j]
        lines = []  # 存储线路分车
        line = [certer_number]  # 每辆车服务客户点，起点是配送中心
        dis_sum = 0  # 线路距离
        dis, d = 0, 0  # 当前客户距离前一个客户的距离、当前客户需求量
        i = 0  # 指向配送中心
        while i < len(bird):
            if line == [certer_number]:  # 车辆未分配客户点
                dis += dis_matrix.loc[certer_number, bird[i]]  # 记录距离
                line.append(bird[i])  # 为客户点分车
                d += Demand[bird[i]]  # 记录需求量
                i += 1  # 指向下一个客户点
            else:  # 已分配客户点则需判断车辆载重和行驶距离
                if (dis_matrix.loc[line[-1], bird[i]] + dis_matrix.loc[bird[i], certer_number] + dis <= DISTABCE) & (
                        d + Demand[bird[i]] <= CAPACITY):
                    dis += dis_matrix.loc[line[-1], bird[i]]
                    line.append(bird[i])
                    d += Demand[bird[i]]
                    i += 1
                else:
                    dis += dis_matrix.loc[line[-1], certer_number]  # 当前车辆装满
                    line.append(certer_number)
                    dis_sum += dis
                    lines.append(line)
                    # 下一辆车
                    dis, d = 0, 0
                    line = [certer_number]

        # 最后一辆车
        dis += dis_matrix.loc[line[-1], certer_number]
        line.append(certer_number)
        dis_sum += dis
        lines.append(line)

        birdPop_car.append(lines)
        fits.append(round(C1 * dis_sum + C0 * len(lines), 1))

    return birdPop_car, fits


def crossover(bird, pLine, gLine, w, c1, c2):
    '''
    采用顺序交叉方式；交叉的parent1为粒子本身，分别以w/(w+c1+c2),c1/(w+c1+c2),c2/(w+c1+c2)
    的概率接受粒子本身逆序、当前最优解、全局最优解作为parent2,只选择其中一个作为parent2；
    输入：bird-粒子,pLine-当前最优解,gLine-全局最优解,w-惯性因子,c1-自我认知因子,c2-社会认知因子；
    输出：交叉后的粒子-croBird；
    '''
    croBird = [None] * len(bird)  # 初始化
    parent1 = bird  # 选择parent1
    # 选择parent2（轮盘赌操作）
    randNum = random.uniform(0, sum([w, c1, c2]))
    if randNum <= w:
        parent2 = [bird[i] for i in range(len(bird) - 1, -1, -1)]  # bird的逆序
    elif randNum <= w + c1:
        parent2 = pLine
    else:
        parent2 = gLine

    # parent1-> croBird
    start_pos = random.randint(0, len(parent1) - 1)
    end_pos = random.randint(0, len(parent1) - 1)
    if start_pos > end_pos: start_pos, end_pos = end_pos, start_pos
    croBird[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()

    # parent2 -> croBird
    list2 = list(range(0, start_pos))
    list1 = list(range(end_pos + 1, len(parent2)))
    list_index = list1 + list2  # croBird从后往前填充
    j = -1
    for i in list_index:
        for j in range(j + 1, len(parent2) + 1):
            if parent2[j] not in croBird:
                croBird[i] = parent2[j]
                break

    return croBird


def draw_path(car_routes, CityCoordinates):
    '''
    #画路径图
    输入：line-路径，CityCoordinates-城市坐标；
    输出：路径图
    '''
    shape = ['o-', '*-', '^-']
    for i in range(len(car_routes)):
        route_i = car_routes[i]
        for route in route_i:
            x, y = [], []
            for i in route:
                Coordinate = CityCoordinates[i]
                x.append(Coordinate[0])
                y.append(Coordinate[1])
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, shape[i], alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    # 车辆参数
    CAPACITY = 120  # 车辆最大容量
    DISTABCE = 250  # 车辆最大行驶距离
    C0 = 30
    C1 = 1

    # PSO参数
    birdNum = 30  # 粒子数量
    w = 0.2  # 惯性因子
    c1 = 0.4  # 自我认知因子
    c2 = 0.4  # 社会认知因子
    pBest, pLine = 0, []  # 当前最优值、当前最优解，（自我认知部分）
    gBest, gLine = 0, []  # 全局最优值、全局最优解，（社会认知部分）

    # 其他参数
    iterMax = 100  # 迭代次数

    bestfit = []  # 记录每代最优值
    DC = 3  # 配送中心个数
    C = 31  # 客户数量

    # 读入数据,Customer0-2表示配送中心，3-33表示配送中心
    Customer = [(50, 25), (25, 75), (75, 75), (96, 24), (40, 5), (49, 8), (13, 7), (29, 89), (48, 30), (84, 39),
                (14, 47), (2, 24), (3, 82), (65, 10), (98, 52), (84, 25), (41, 69), (1, 65),
                (51, 71), (75, 83), (29, 32), (83, 3), (50, 93), (80, 94), (5, 42), (62, 70), (31, 62), (19, 97),
                (91, 75), (27, 49), (23, 15), (20, 70), (85, 60), (98, 85)]
    Demand = [0, 0, 0, 16, 11, 6, 10, 7, 12, 16, 6, 16, 8, 14, 7, 16, 3, 22, 18, 19, 1, 14, 8, 12, 4, 8, 24, 24, 2, 10,
              15, 2, 14, 9]

    dis_matrix = calDistance(Customer)  # 计算城市间距离

    # 分配客户点到配送中心
    distribution_centers = assign_distribution_center(dis_matrix, DC, C)
    bestfit_list, gLine_car_list = [], []

    for certer_number in range(len(distribution_centers)):
        distribution_center = distribution_centers[certer_number]
        birdPop = [greedy(distribution_center, dis_matrix, DC, certer_number) for i in range(birdNum)]  # 贪婪算法构造初始解
        birdPop_car, fits = calFitness(birdPop, certer_number, Demand, dis_matrix, CAPACITY, DISTABCE, C0,
                                       C1)  # 分配车辆，计算种群适应度

        gBest = pBest = min(fits)  # 全局最优值、当前最优值
        gLine = pLine = birdPop[fits.index(min(fits))]  # 全局最优解、当前最优解
        gLine_car = pLine_car = birdPop_car[fits.index(min(fits))]

        iterI = 1  # 当前迭代次数
        while iterI <= iterMax:  # 迭代开始
            for i in range(birdNum):
                birdPop[i] = crossover(birdPop[i], pLine, gLine, w, c1, c2)

            birdPop_car, fits = calFitness(birdPop, certer_number, Demand, dis_matrix, CAPACITY, DISTABCE, C0,
                                           C1)  # 分配车辆，计算种群适应度
            pBest, pLine, pLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]
            if min(fits) <= gBest:
                gBest, gLine, gLine_car = min(fits), birdPop[fits.index(min(fits))], birdPop_car[fits.index(min(fits))]

            print(iterI, gBest)  # 打印当前代数和最佳适应度值
            iterI += 1  # 迭代计数加一

        bestfit_list.append(gBest)
        gLine_car_list.append(gLine_car)

    print(gLine_car_list)  # 路径顺序
    print("最优值：", sum(bestfit_list))
    draw_path(gLine_car_list, Customer)  # 画路径图
