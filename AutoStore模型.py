l = 600  # 货格长度
w = 400  # 货格宽度
h = 7000  # 货格高度

# 商品货格范围
com_range = {'green': [(1, 4), (1, 5)], 'blue': [(5, 7), (1, 10)], 'red': [(8, 10), (1, 4)],
             'yellow': [(8, 10), (5, 10)], 'pink': [(1, 4), (6, 10)]}  # [列,行]

dingdan = {'red': 5, 'yellow': 5, 'green': 4, 'blue': 3, 'pink': 3}

store = []
m = 10  # 货格横向个数
n = 10  # 货格纵向个数
for i in range(1, m + 1):
    temp = []
    for j in range(1, n + 1):
        temp.append((i, j))
    store.append(temp)

cap = [
    [20, 18, 18, 20, 18, 20, 20, 20, 9, 10],
    [20, 20, 19, 18, 18, 19, 20, 18, 9, 10],
    [19, 20, 19, 19, 19, 19, 19, 19, 9, 10],
    [20, 18, 18, 19, 20, 19, 19, 18, 9, 8],
    [20, 20, 19, 20, 19, 20, 20, 19, 10, 9],
    [20, 20, 20, 20, 20, 18, 18, 18, 8, 10],
    [19, 19, 18, 18, 20, 19, 19, 20, 8, 10],
    [18, 18, 19, 19, 20, 19, 20, 19, 10, 9],
    [20, 19, 19, 20, 19, 18, 19, 20, 10, 10],
    [19, 19, 19, 19, 20, 18, 20, 19, 9, 10],
]  # 从下往上数

store_dict = []  # 描述仓储货格的数据结构（未区分各种货物）
for row in range(m):
    s_line = store[row]
    c_line = cap[row]
    new_row = []
    for index in range(len(s_line)):
        temp = {}
        temp[s_line[index]] = c_line[index]
        new_row.append(temp)
    store_dict.append(new_row)


def distance(dingdan, store_dict):  # 传入订单为一个字典
    init_h = 20  # 初始货格高度
    init_h_t = 10
    init = (1, 0)  # 小车初始位置
    things = dingdan.items()
    for thing in things:
        area = com_range[thing[0]]
        n = thing[1]
        """
        需要截取二维数组的一个区域，循环n次
        """
        for _ in range(n):  # 执行n次订单
            min = 999999
            for row in store_dict[area[1][0] - 1:area[1][1]]:  # 取合适的行区域
                for dict in row[area[0][0] - 1:area[0][1]]:  # 取合适的列区域
                    loc = list(dict.keys())[0]
                    cap = list(dict.values())[0]
                    if cap <= 10:
                        dis = abs(loc[0] - init[0]) * w + abs(loc[1] - init[1]) * l + 290 * (init_h_t - cap)
                    else:
                        dis = abs(loc[0] - init[0]) * w + abs(loc[1] - init[1]) * l + (h - 290 * cap)
                    if dis <= min:
                        min = dis
                        best_loc = loc
                        temp_cap = cap
            store_dict[best_loc[0] - 1][best_loc[1] - 1][best_loc] = temp_cap - 1  # 更新货物数量/高度
            a = store_dict[best_loc[0] - 1][best_loc[1] - 1]
            """
            需要输出出货的位置
            """
            print(best_loc)


if __name__ == "__main__":
    distance(dingdan, store_dict)
