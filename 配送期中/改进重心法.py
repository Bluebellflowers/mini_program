# coding: utf-8
from math import sqrt

# ## 重力法求仓库最佳选址

# w：物流量
# h：运输费率
# p：仓库坐标
w = [2000, 3000, 2500, 1000, 1500]
h = [0.05, 0.05, 0.075, 0.075, 0.075]
p = [[3, 8], [8, 2], [2, 5], [6, 10], [8, 8]]
condition = 0.1


def improved_center_of_gravity_method(w, h, p, condition=0.05):
    # 计算x0,y0
    sum_x0 = 0
    sum_x0_y = 0
    for i in range(len(w)):
        sum_x0 += h[i] * w[i] * p[i][0]
        sum_x0_y += h[i] * w[i]
    x0 = sum_x0 / sum_x0_y

    sum_y0 = 0
    sum_y0_y = 0
    for i in range(len(w)):
        sum_y0 += h[i] * w[i] * p[i][1]
        sum_y0_y += h[i] * w[i]
    y0 = sum_y0 / sum_y0_y

    num = 1
    while True:
        # 储存上一次计算结果
        x_pre = x0
        y_pre = y0
        # 计算当前x
        sum_x0 = 0
        sum_x0_y = 0
        for i in range(len(w)):
            # 计算d
            d1 = sqrt((x0 - p[i][0]) ** 2 + (y0 - p[i][1]) ** 2)
            sum_x0 += h[i] * w[i] * p[i][0] / d1
            sum_x0_y += h[i] * w[i] / d1
        # 计算当前y
        sum_y0 = 0
        sum_y0_y = 0
        for i in range(len(w)):
            d2 = sqrt((x0 - p[i][0]) ** 2 + (y0 - p[i][1]) ** 2)
            sum_y0 += h[i] * w[i] * p[i][1] / d2
            sum_y0_y += h[i] * w[i] / d2
        x0 = sum_x0 / sum_x0_y
        y0 = sum_y0 / sum_y0_y
        if __name__ == '__main__':
            print("第{}次迭代结果如下：".format(num))
            print("仓库坐标为:", x0, y0)
        error = sqrt((x_pre - x0) ** 2 + (y_pre - y0) ** 2)
        if __name__ == '__main__':
            print("误差为：", error)
        # 终止条件
        if error < condition:
            break
        num += 1
    return x0, y0


if __name__ == '__main__':
    x, y = improved_center_of_gravity_method(w, h, p, condition)
    X = [x, y]  # 坐标


    def get_dis(X):  # 计算需求点与配送中心间的距离函数
        dis = []
        for val in p:
            dis_part = sqrt((X[0] - val[0]) ** 2 + (X[1] - val[1]) ** 2)
            dis.append(dis_part)
        return dis


    dis = get_dis(X)
    u = 0  # 改进前的总费用
    for area in range(len(p)):
        ug = w[area] * h[area] * dis[area]
        u += ug
    print('目标函数值为：{}'.format(u))
