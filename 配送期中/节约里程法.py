import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
from copy import  deepcopy

dist = [
    [10],
    [9, 4],
    [7, 9, 5],
    [8, 14, 10, 5],
    [8, 18, 14, 9, 6],
    [8, 18, 17, 15, 13, 7],
    [3, 13, 12, 10, 11, 10, 6],
    [4, 14, 13, 11, 12, 12, 8, 2],
    [10, 11, 15, 17, 18, 18, 17, 11, 9],
    [7, 4, 8, 13, 15, 15, 15, 10, 11, 8]
]

# =============================将当前三角矩阵转置方便合并成距离矩阵=========================================
dist_transpose = []
for i in range(len(dist)):
    temp = []
    for j in range(1, len(dist[-1]) - i + 1):
        temp.append(dist[-1 * j][i])
    dist_transpose.append(temp[::-1])
# 其余位置填充0元素
for i in range(len(dist_transpose)):
    for j in range(i + 1):
        dist_transpose[i].insert(0, 0)
dist_transpose.append([0 for i in range(len(dist_transpose[0]))])
# 其余位置填充0元素
for i in range(len(dist)):
    for j in range(len(dist) - i):
        dist[i].append(0)
dist.insert(0, [0 for i in range(len(dist[0]))])
# 合并上三角矩阵和下三角矩阵为距离矩阵
dist = np.array(dist)
dist_transpose = np.array(dist_transpose)
dist_total = dist + dist_transpose
dist = dist_total
# ====================================参数设置=============================================
demands = np.array([0, 140, 160, 40, 40, 40, 150, 60, 160, 120, 50])  # 需求
cap = 400  # 车辆容量
n_customers = len(demands) - 1  # 客户数量
# ====================================初始化图===========================================
G = nx.DiGraph()
G.add_node(0)  # 添加中心节点
G.add_nodes_from([i for i in range(1, n_customers + 1)])  # 添加客户结点
# 初始化图，是每个客户结点与中心点连接
n_routes = n_customers
G.add_edges_from([(0, i, {'weight': dist[0, i]}) for i in range(1, n_customers + 1)])
G.add_edges_from([(i, 0, {'weight': dist[i, 0]}) for i in range(1, n_customers + 1)])

# 绘图参数的设置
pos = nx.spring_layout(G)
nx.draw(G, pos, font_size=16, with_labels=False)
# 绘制边标签
nx.draw_networkx_labels(G, pos)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# 计算出各点之间的节约里程
saving = []
for i in range(1, n_customers + 1):
    for j in range(i + 1, n_customers + 1):
        if i != j:
            # 计算节约历程的公式a+b-c
            temp_distance = dist[i][0] + dist[0][j] - dist[i][j]
            if temp_distance > 0:
                saving.append(((i, j), temp_distance))
# 根据temp_distance排序
saving = sorted(saving, key=itemgetter(1), reverse=True)

for i in range(len(saving)):
    sav = saving[i][0]
    # judge1 = True
    #
    # # i，j不在同一路径
    # for route in nx.simple_cycles(G):
    #     route.append(0)
    #     # print(route)
    #     if sav[0] in route and sav[1] in route:
    #         judge1 = False
    #         break
    #
    #  # i，j 都在路径
    # judge2 = False
    #
    # for route in nx.simple_cycles(G):
    #     route.append(0)
    #     if route[1] == sav[0]:
    #         judge2 = True
    #         break

    judge1 = True
    judge2 = True

    # 判断是否和自己成环
    judge3 = False
    for route in nx.simple_cycles(G):
        route.append(0)
        if route[-2] != sav[1]:
            judge3 = True
            break

    # 容量检查
    judge4 = True
    for route in nx.simple_cycles(G):
        if sav[0] in route:
            sav0_route_dem = (sum(demands[route]))
        if sav[1] in route:
            sav1_route_dem = (sum(demands[route]))
    if sav0_route_dem + sav1_route_dem > cap:
        judge4 = False

    # 条件都满足
    if judge1 and judge2 and judge3 and judge4:
        if G.has_edge(sav[0], 0) & G.has_edge(0, sav[1]):
            G.remove_edge(sav[0], 0)
            G.remove_edge(0, sav[1])
            G.add_edge(sav[0], sav[1], weight=dist[sav[0]][sav[1]])
            n_routes -= 1
        elif G.has_edge(0, sav[0]) & G.has_edge(sav[1], 0):
            G.remove_edge(0, sav[0])
            G.remove_edge(sav[1], 0)
            G.add_edge(sav[1], sav[0], weight=dist[sav[1]][sav[0]])
            n_routes -= 1

pos = nx.spring_layout(G)
nx.draw(G, pos, font_size=16, with_labels=False)
nx.draw_networkx_labels(G, pos)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# 对结果进行分析
g_info=G.adj
# 初始结点为0即为一条运输路线
init_node=g_info[0]
route_node=deepcopy(g_info)
total_route=[]
weight=0  # 记录路线长度
# 导出运输路线
for node in init_node:
    part_route = [0]  # 记录路线
    tail=[node]
    weight+=init_node[node]['weight']
    part_route.extend(tail)
    while True:
        now_dict=route_node[tail[0]]  # 存储当前路线情况的字典
        tail = list(route_node[tail[0]].keys())  # 该条线路的尾结点
        if tail[0]==0:
            part_route.append(0)
            weight += now_dict[tail[0]]['weight']
            total_route.append(np.array(part_route))
            break
        else:
            part_route.extend(tail)
            weight += now_dict[tail[0]]['weight']
total_route=np.array(total_route)
total_route=1+total_route
for index,_ in enumerate(total_route):
    print("第{}调路线为：".format(index+1))
    for i in _:
        print('{}-->'.format(i),end="")
    print()
print('路线长度为：{}'.format(weight))