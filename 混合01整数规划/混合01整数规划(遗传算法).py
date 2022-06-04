import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv


class Sol():
    # 一种解
    def __init__(self):
        self.node_id_list = None
        self.obj = None
        self.fit = None
        self.routes = None  # 路线
        self.supplies = None  # 对应的物资调度
        self.meet = None  # 满足度


class Node():
    # 节点
    def __init__(self):
        self.id = 0  # 序号
        self.x_coord = 0  # 横坐标
        self.y_coord = 0  # 纵坐标
        self.demand = 0  # 需求
        self.level = 0  # 等级
        self.depot_capacity = 15  # 容量


class Model():
    def __init__(self):
        self.best_sol = None
        self.demand_dict = {}  # 受灾点
        self.depot_dict = {}  # 配送中心
        self.demand_id_list = []  # 需求点集合
        self.sol_list = []  # 解集
        self.sol_list_supplies = []  # 对应的物资集
        self.distance_matrix = {}  # 距离矩阵
        self.opt_type = 0
        self.vehicle_cap = 0  # 车辆容量
        self.pc = 0.5  # 交叉 概率
        self.pm = 0.2  # 变异
        self.n_select = 20  # 优良个体选择数量
        self.popsize = 30  # 种群


def readCsvFile(demand_file, depot_file, model):
    """
    读取文件
    """
    # 灾害点
    # 各个等级的满意度
    satisfaction = {'1': 0.9, '2': 0.8, '3': 0.7}

    with open(demand_file, 'r') as f:
        demand_reader = csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])  # 横坐标
            node.y_coord = float(row['y_coord'])  # 纵坐标
            node.demand = float(row['demand'])  # 需求
            node.level = satisfaction[str(row['level'])]  # 等级对应的最小满意度
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)
    # 配送点
    with open(depot_file, 'r') as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.depot_capacity = float(row['capacity'])
            model.depot_dict[node.id] = node


def calDistance(model):
    """
    计算距离
    distance_matrix属性储存任意节点间的欧式距离；
    """
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
        for _, depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - depot.y_coord) ** 2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist


def selectDepot(route, supply, depot_dict, model):
    """
    选择配送中心
    分配最近配送中心作为其物资提供基地
    """
    min_in_out_distance = float('inf')
    index = None
    for _, depot in depot_dict.items():
        # 对每个配送中心
        if depot.depot_capacity > 0:  # 剩余物资大于0
            in_out_distance = model.distance_matrix[depot.id, route[0]] + model.distance_matrix[route[-1], depot.id]
            if in_out_distance < min_in_out_distance:  # 小于最小距离
                index = depot.id  # 更新
                min_in_out_distance = in_out_distance
    if index is None:
        print("there is no vehicle to dispatch")
    route.insert(0, index)
    route.append(index)
    depot_dict[index].depot_capacity = depot_dict[index].depot_capacity - sum(supply)
    return route, depot_dict, min_in_out_distance


def splitRoutes(node_id_list, model):
    """
    路径选择
    """
    num_vehicle = 0  # 当前解需要的车辆数
    level_satisfaction = 0  # 满足度

    all_distance = 0  # 距离

    vehicle_routes = []  # 路线
    route = []  # 一段的解

    all_supplies = []  # 总物资
    supply = []  # 一段的物资

    remained_cap = model.vehicle_cap
    depot_dict = copy.deepcopy(model.depot_dict)
    for node_id in node_id_list:
        # 选择路径了

        # 配送量 满足最小基础上加一点 不超过总需求
        tmp = model.demand_dict[node_id].demand * model.demand_dict[node_id].level + random.randint(0, int(
            model.demand_dict[node_id].demand * (1 - model.demand_dict[node_id].level)))
        if remained_cap - tmp >= 0:  # 能送
            # 还能装
            route.append(node_id)  # 路线
            supply.append(tmp)  # 对应的物资

            # 没满足的比例
            level_satisfaction += (model.demand_dict[node_id].demand - tmp) / model.demand_dict[node_id].demand
            # 剩余容量
            remained_cap = remained_cap - tmp
        else:
            # 不能装 就开辟新的！
            route, depot_dict, min_in_out_distance = selectDepot(route, supply, depot_dict, model)
            all_distance += min_in_out_distance
            vehicle_routes.append(route)
            all_supplies.append(supply)  # 汇总

            route = [node_id]  # 新的
            supply = [tmp]
            level_satisfaction += (model.demand_dict[node_id].demand - tmp) / model.demand_dict[node_id].demand

            num_vehicle = num_vehicle + 1
            # 剩余容量
            remained_cap = model.vehicle_cap - tmp
    # print('一段！',route)
    # 选择配送中心
    route, depot_dict, min_in_out_distance = selectDepot(route, supply, depot_dict, model)
    all_distance += min_in_out_distance
    vehicle_routes.append(route)
    all_supplies.append(supply)  # 汇总
    return num_vehicle, level_satisfaction, vehicle_routes, all_supplies, all_distance


def calRouteDistance(route, model):
    distance = 0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        distance += model.distance_matrix[from_node, to_node]
    return distance


def calFit(model):
    # 计算适应度
    # calculate fit value：fit=Objmax-obj
    max_obj = -float('inf')
    best_sol = Sol()  # record the local best solution
    best_sol.obj = float('inf')

    for sol in model.sol_list:
        node_id_list = sol.node_id_list
        num_vehicle, level_satisfaction, vehicle_routes, all_supplies, all_distance = splitRoutes(node_id_list, model)

        if model.opt_type == 0:  # 组合解！！三个约束的组合
            # 归一化
            num_vehicle_st = abs(num_vehicle - 7) / 15
            level_satisfaction_st = abs(level_satisfaction - 6) / 9
            #  print('xx',all_distance)
            all_distance_st = abs(all_distance - 300) / 550
            # 目标
            sol.obj = 0.5 * all_distance_st + 0.3 * level_satisfaction + 0.2 * num_vehicle_st

            sol.routes = vehicle_routes
            sol.supplies = all_supplies
            sol.meet = level_satisfaction

            if sol.obj > max_obj:
                max_obj = sol.obj
            if sol.obj < best_sol.obj:
                best_sol = copy.deepcopy(sol)
        else:
            distance = 0
            for route in vehicle_routes:
                distance += calRouteDistance(route, model)
            sol.obj = distance
            sol.routes = vehicle_routes
            sol.supplies = all_supplies
            sol.meet = level_satisfaction
            if sol.obj > max_obj:
                max_obj = sol.obj
            if sol.obj < best_sol.obj:
                best_sol = copy.deepcopy(sol)

    # 差距
    for sol in model.sol_list:
        sol.fit = max_obj - sol.obj
    # 更新
    if best_sol.obj < model.best_sol.obj:  # 最小的目标！！

        model.best_sol = best_sol


def generateInitialSol(model):
    """
    初始化 随机种群
    """
    demand_id_list = copy.deepcopy(model.demand_id_list)
    for i in range(model.popsize):
        seed = int(random.randint(0, 10))  # 随机种子
        random.seed(seed)
        random.shuffle(demand_id_list)  # 打乱
        sol = Sol()  # 解
        sol.node_id_list = copy.deepcopy(demand_id_list)  # 解集
        model.sol_list.append(sol)  # 添加到模型里


# Binary tournament
def selectSol(model):
    """
    物竞天择 适者生存
    根据适应度淘汰个体
    """
    sol_list = copy.deepcopy(model.sol_list)
    model.sol_list = []
    for i in range(model.n_select):
        f1_index = random.randint(0, len(sol_list) - 1)
        f2_index = random.randint(0, len(sol_list) - 1)
        f1_fit = sol_list[f1_index].fit
        f2_fit = sol_list[f2_index].fit
        if f1_fit < f2_fit:
            model.sol_list.append(sol_list[f2_index])
        else:
            model.sol_list.append(sol_list[f1_index])


def crossSol(model):
    """
    交叉
    """
    sol_list = copy.deepcopy(model.sol_list)
    model.sol_list = []
    while True:
        f1_index = random.randint(0, len(sol_list) - 1)
        f2_index = random.randint(0, len(sol_list) - 1)
        if f1_index != f2_index:
            f1 = copy.deepcopy(sol_list[f1_index])
            f2 = copy.deepcopy(sol_list[f2_index])
            if random.random() <= model.pc:
                cro1_index = int(random.randint(0, len(model.demand_id_list) - 1))
                cro2_index = int(random.randint(cro1_index, len(model.demand_id_list) - 1))
                new_c1_f = []
                new_c1_m = f1.node_id_list[cro1_index:cro2_index + 1]
                new_c1_b = []
                new_c2_f = []
                new_c2_m = f2.node_id_list[cro1_index:cro2_index + 1]
                new_c2_b = []
                for index in range(len(model.demand_id_list)):
                    if len(new_c1_f) < cro1_index:
                        if f2.node_id_list[index] not in new_c1_m:
                            new_c1_f.append(f2.node_id_list[index])
                    else:
                        if f2.node_id_list[index] not in new_c1_m:
                            new_c1_b.append(f2.node_id_list[index])
                for index in range(len(model.demand_id_list)):
                    if len(new_c2_f) < cro1_index:
                        if f1.node_id_list[index] not in new_c2_m:
                            new_c2_f.append(f1.node_id_list[index])
                    else:
                        if f1.node_id_list[index] not in new_c2_m:
                            new_c2_b.append(f1.node_id_list[index])
                new_c1 = copy.deepcopy(new_c1_f)
                new_c1.extend(new_c1_m)
                new_c1.extend(new_c1_b)
                f1.nodes_seq = new_c1
                new_c2 = copy.deepcopy(new_c2_f)
                new_c2.extend(new_c2_m)
                new_c2.extend(new_c2_b)
                f2.nodes_seq = new_c2
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            else:
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            if len(model.sol_list) > model.popsize:
                break


def muSol(model):
    """
    突变
    """
    sol_list = copy.deepcopy(model.sol_list)
    model.sol_list = []
    while True:
        f1_index = int(random.randint(0, len(sol_list) - 1))
        f1 = copy.deepcopy(sol_list[f1_index])
        m1_index = random.randint(0, len(model.demand_id_list) - 1)
        m2_index = random.randint(0, len(model.demand_id_list) - 1)
        if m1_index != m2_index:
            if random.random() <= model.pm:
                node1 = f1.node_id_list[m1_index]
                f1.node_id_list[m1_index] = f1.node_id_list[m2_index]
                f1.node_id_list[m2_index] = node1
                model.sol_list.append(copy.deepcopy(f1))
            else:
                model.sol_list.append(copy.deepcopy(f1))
            if len(model.sol_list) > model.popsize:
                break


def plotObj(obj_list):
    """
    绘图
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1, len(obj_list) + 1), obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1, len(obj_list) + 1)
    plt.savefig('迭代图像.png')
    plt.show()


def outPut(model, times):
    """
    将结果保存到文件
    """
    work = xlsxwriter.Workbook(f'result{times}.xlsx')
    worksheet = work.add_worksheet()
    worksheet.write(0, 0, '')
    worksheet.write(1, 0, '路线')
    if model.opt_type == 0:  # 组合解
        worksheet.write(0, 1, '三目标最优')
    else:
        worksheet.write(0, 1, '最短时间')
    worksheet.write(0, 2, '平均不满足程度')
    worksheet.write(0, 3, '总物资量')
    worksheet.write(1, 1, model.best_sol.obj)  # 目标值
    worksheet.write(1, 2, model.best_sol.meet)  # 平均不满足程度
    sum_sp = 0  # 总物资
    for row, supply in enumerate(model.best_sol.supplies):  # 物资
        s = [int(i) for i in supply]  # 对应的物资
        sum_sp += sum(s)

    worksheet.write(1, 3, sum_sp)  # 总物资

    for row, route in enumerate(model.best_sol.routes):  # 路线
        worksheet.write(row + 2, 0, 'v' + str(row + 1))
        r = [str(i) for i in route]  # 路线
        worksheet.write(row + 2, 1, '-'.join(r))

    for row, supply in enumerate(model.best_sol.supplies):  # 物资

        s = [str(i) for i in supply]  # 对应的物资
        worksheet.write(row + 2, 3, '-'.join(s))

    work.close()


def plotRoutes(model, times):
    for route in model.best_sol.routes:
        x_coord = [model.depot_dict[route[0]].x_coord]
        y_coord = [model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0] == 'd1':
            plt.plot(x_coord, y_coord, marker='o', color='#DC143C', linewidth=0.5, markersize=5)
        elif route[0] == 'd2':
            plt.plot(x_coord, y_coord, marker='o', color='#000000', linewidth=0.5, markersize=5)
        elif route[0] == 'd3':
            plt.plot(x_coord, y_coord, marker='o', color='#00FF7F', linewidth=0.5, markersize=5)
        elif route[0] == 'd4':
            plt.plot(x_coord, y_coord, marker='o', color='#4169E1', linewidth=0.5, markersize=5)
        elif route[0] == 'd5':
            plt.plot(x_coord, y_coord, marker='o', color='#00FFFF', linewidth=0.5, markersize=5)
        elif route[0] == 'd6':
            plt.plot(x_coord, y_coord, marker='o', color='#32CD32', linewidth=0.5, markersize=5)

        # 配送点
        df_test = pd.read_csv('配送中心.csv')
        for i in range(len(df_test)):
            if df_test.iloc[i, 0] == 'd1':
                plt.scatter(df_test.iloc[i, 1], df_test.iloc[i, 2], s=100, marker='*', color='#DC143C')
            if df_test.iloc[i, 0] == 'd2':
                plt.scatter(df_test.iloc[i, 1], df_test.iloc[i, 2], s=100, marker='*', color='#000000')
            if df_test.iloc[i, 0] == 'd3':
                plt.scatter(df_test.iloc[i, 1], df_test.iloc[i, 2], s=100, marker='*', color='#00FF7F')
            if df_test.iloc[i, 0] == 'd4':
                plt.scatter(df_test.iloc[i, 1], df_test.iloc[i, 2], s=100, marker='*', color='#4169E1')
            if df_test.iloc[i, 0] == 'd5':
                plt.scatter(df_test.iloc[i, 1], df_test.iloc[i, 2], s=100, marker='*', color='#00FFFF')

    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.savefig(f'路线图{times}.png')
    plt.show()


def Run(demand_file, distribution_file, epochs, pc, pm, popsize, n_select, v_cap, opt_type, times=1):
    """
受灾文件
配送中心
迭代次数
交叉概率
变异概率
种群规模
优良个体选择数量
单次容量
    """
    model = Model()
    model.vehicle_cap = v_cap  # 单次容量
    model.opt_type = opt_type  # 优化目标类型，0：组合目标，1：最小时间
    model.pc = pc  # 交叉 概率
    model.pm = pm  # 突变
    model.popsize = popsize  # 种群规模
    model.n_select = n_select  # 优良个体选择数量

    readCsvFile(demand_file, distribution_file, model)  ##读取两个文件！ 把点存进去
    calDistance(model)  # 得到距离
    generateInitialSol(model)  ##初始解生成
    history_best_obj = []  # 历史最优解
    best_sol = Sol()  # 假定一个最优解 贪心
    best_sol.obj = float('inf')
    model.best_sol = best_sol
    for ep in range(epochs):  # 迭代
        # 每个个体都是一个解 一代有100个解
        calFit(model)
        selectSol(model)
        crossSol(model)
        muSol(model)
        history_best_obj.append(model.best_sol.obj)
    #     print("%s/%s， best obj: %s" % (ep,epochs,model.best_sol.obj))
    plotObj(history_best_obj)
    plotRoutes(model, times)
    outPut(model, times)
    return model.best_sol.obj


if __name__ == '__main__':
    demand_file = '受灾点.csv'  # 受灾点 坐标、需求量 、等级
    depot_file = '配送中心.csv'  # 配送中心
    # 各个等级的满意度
    #  satisfaction = {'1':0.9,'2':0.8,'3':0.7}
    # 求解
    best_obj = 100  # 最优
    for times in range(10):
        seed = int(random.randint(0, 10))  # 随机种子
        obj = Run(demand_file, depot_file, epochs=25000, pc=0.5, pm=0.5, popsize=30, n_select=80, v_cap=60, opt_type=0,
                  times=times)
        #   print('xx')
        if obj < best_obj:  # 更新
            best_obj = obj
            print(times)

# 绘图
import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt

# 应用实际坐标点
df_true1 = pd.read_csv('实际配送点.csv')
df_true1 = df_true1.iloc[:5, :]
# 应用实际坐标点
df_true2 = pd.read_csv('实际受灾点(30).csv')
szd = []
for i in range(len(df_true2)):
    wd = round(float(df_true2['纬度'][i]), 5)
    jd = round(float(df_true2['经度'][i]), 5)
    szd.append([wd, jd])

psd = []
for i in range(len(df_true1)):
    wd = round(float(df_true1['纬度'][i]), 5)
    jd = round(float(df_true1['经度'][i]), 5)
    psd.append([wd, jd])

# ## 初始化地图，指定成都
# m = folium.Map(
#     location=[30.605604, 104.065846],
#     zoom_start=8
# )
# groups = folium.FeatureGroup('momomomo')
#
# locations = szd
# # 受灾点
# for l in locations:
#     groups.add_child(
#         folium.Marker(
#             location=l,
#         )
#     )
# # 配送点
# locations = psd
# for l in locations:
#     groups.add_child(
#         folium.CircleMarker(
#             location=l,
#             radius=9,
#             color='green',
#             fill=True,
#             fill_color='green',
#             fill_opacity=0.4
#         )
#     )
#
# m.add_child(groups)
# folium.LayerControl().add_to(m)
