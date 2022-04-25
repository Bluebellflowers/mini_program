from 改进重心法 import improved_center_of_gravity_method
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from math import sqrt


class Liqun:
    def __init__(self, nw, p, w, h, xrange, yrange):
        # Pso参数：
        # nr 配送中心个数
        self.ww = 0.75  # 惯性因子
        self.c1 = 0.2  # 学习因子 1
        self.c2 = 0.2  # 学习因子 2
        self.dim = 2 * nw  # 维度的维度#对应 2 个参数 x,y (针对具体问题）
        self.size = 100  # 种群大小，即种群中小鸟的个数
        self.iter_num = 50  # 算法最大迭代次数
        self.max_vel = 0.5  # 限制粒子的最大速度为 0.5
        self.min_vel = -0.5  # 限制粒子的最小速度为-0.5
        self.p = p  # 坐标矩阵
        self.w = w  # 需求量列表
        self.h = h  # 运输费率
        self.xrange = xrange
        self.yrange = yrange

    # ============================================ 目标函数公式：===================================================
    def calc_f(self, X):  # 计算个体粒子的的适应度值，也就是目标函数值，X 的维度是 size * dim
        # X由wn个x坐标和wn个y坐标排列而成
        def division(X):  # 区域划分函数
            x = X[:int(self.dim / 2)]
            y = X[int(self.dim / 2):]
            Q=[]
            for _ in range(int(self.dim / 2)):
                Q.append([])  # 配送中心区域集合
            for pi, pp in enumerate(self.p):
                dis_total = []
                for index in range(len(x)):  # 对每一个配送中心计算一次
                    temp = sqrt((x[index] - pp[0]) ** 2 + (y[index] - pp[1]) ** 2)
                    dis_total.append(temp)
                dis_min = min(dis_total)
                dis_min_area = dis_total.index(dis_min)
                Q[dis_min_area].append(pi)  # 将需求点序号加入某个配送中心的区域
            return Q

        def get_division_judge(X):  # 获得区域划分并防止退化函数
            Q = division(X)  # 进行第一次计算
            judge = True
            while judge:
                num = 0
                for _ in Q:
                    if len(_) == 0:  # 若有一个区域需求点为0，即发生退化，需重新生成点
                        judge = True
                        X = np.random.uniform(3, 8, size=(int(self.dim / 2)))
                        X = np.hstack([X, np.random.uniform(2, 10, size=(int(self.dim / 2)))])
                        break
                    else:
                        num += 1
                if num == len(Q):
                    break
                Q = division(X)
            return X, Q

        """接下来进行重心法的迭代"""
        X, Q = get_division_judge(X)

        def get_point(part, item):
            p_cal = []  # 当前区域内的需求点
            item = item
            for index in part:
                p_cal.append(item[index])
            return p_cal

        x1 = X[:int(self.dim / 2)]
        y1 = X[int(self.dim / 2):]
        location = []
        for index,part in enumerate(Q):  # 循环当前区域内的需求点和获得相关数据计算改进后的中心店
            p_c = get_point(part, self.p)
            w_c = get_point(part, self.w)
            h_c = get_point(part, self.h)
            if len(part)==1:
                location.append([x1[index], y1[index]])
            else:
                x0, y0 = improved_center_of_gravity_method(w_c, h_c, p_c)
                location.append([x0, y0])

        # 将坐标转换成和输入形式一致得到坐标方便调用division
        xx = []
        yy = []
        for part in location:
            xx.append(part[0])
            yy.append(part[1])
        xx.extend(yy)
        X_pro = xx
        # 计算重心法求得配送中心新的区域划分以及退化检验
        X_pro, Q_pro = get_division_judge(X_pro)  # 新的分区

        def get_dis(X, Q):  # 计算需求点与配送中心间的距离函数
            x = X[:int(self.dim / 2)]
            y = X[int(self.dim / 2):]
            d_c = [None] * len(self.p)
            for index in range(len(x)):
                for part1 in Q:
                    for r_index in part1:
                        dis = sqrt((x[index] - self.p[r_index][0]) ** 2 + (y[index] - self.p[r_index][1]) ** 2)
                        d_c[r_index] = dis
            return d_c

        # 计算费用
        X_d = get_dis(X, Q)
        Q_u = 0  # 改进前的总费用
        for part in Q:
            for area in part:
                ug = self.w[area] * self.h[area] * X_d[area]
                Q_u += ug
        X_pro_d = get_dis(X_pro, Q_pro)
        Q_pro_u = 0  # 改进后的总费用
        for part in Q_pro:
            for area in part:
                ug = self.w[area] * self.h[area] * X_pro_d[area]
                Q_pro_u += ug

        if Q_pro_u < Q_u:  # 迭代终止条件
            X = X_pro
            f = Q_pro_u
            return X, f
        else:
            f = Q_u
            return X, f

    # =====================================================粒子群速度更新公式：============================================
    def velocity_update(self, V, X, pbest, gbest):
        # 根据速度更新公式更新每个粒子的速度
        # 种群 size
        # V: 粒子当前的速度矩阵，size*dim 的矩阵
        # X: 粒子当前的位置矩阵，size*dim 的矩阵
        # pbest: 每个粒子历史最优位置，size*dim 的矩阵
        # gbest: 种群历史最优位置，1*dim 的矩阵
        r1 = np.random.random((self.size, 1))
        r2 = np.random.random((self.size, 1))
        V = self.ww * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)  # 直接对照公式写就好了

        # 防止越界：
        V[V > self.max_vel] = self.max_vel
        V[V < self.min_vel] = self.min_vel
        return V

    # ===================================================粒子群位置更新公式：================================================
    def position_update(self, X, V):
        # 根据公式更新粒子的位置
        # X: 粒子当前的位置矩阵，维度是 size*dim
        # V: 粒子当前的速度矩阵，维度是 size*dim
        X = X + V
        size = np.shape(X)[0]
        # 判断位置更新后粒子是否超过边界,若超过边界则取边界值
        """需要重写"""
        for i in range(size):
            for j in range(int(self.dim / 2)):
                if X[i][j] <= self.xrange[0]:
                    X[i][j] = self.xrange[0]
                if X[i][j] >= self.xrange[1]:
                    X[i][j] = self.xrange[1]
            for j in range(int(self.dim / 2), self.dim):
                if X[i][j] <= self.yrange[0]:
                    X[i][j] = self.yrange[0]
                if X[i][j] >= self.yrange[1]:
                    X[i][j] = self.yrange[1]
        return X

    # ========================================================= 更新种群函数：==============================================
    def update_pbest(self, pbest, pbest_fitness, xi, xi_fitness):
        # 判断是否需要更新粒子的历史最优位置
        # pbest: 历史最优位置
        # pbest_fitness: 历史最优位置对应的适应度值
        # xi: 当前位置
        # xi_fitness: 当前位置的适应度函数值
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness
        else:
            return xi, xi_fitness

    # ====================================================更新全局最优位置==================================================
    def update_gbest(self, gbest, gbest_fitness, pbest, pbest_fitness):
        for index,var in enumerate(pbest_fitness):
            if var<gbest_fitness:
                gbest_fitness=var
                gbest=pbest[index]
        return gbest, gbest_fitness

    # ========================================================== 主函数===================================================
    def main(self):
        # 初始化一个矩阵 info, 记录：
        # 0、种群每个粒子的历史最优位置对应的适应度，
        # 1、当前适应度，
        info = np.zeros((self.size, 2))
        fitneess_value_list = []  # 记录每次迭代过程中的种群适应度值变化
        # ====================================================初始化===========================================================
        # 用一个 size*dim 的矩阵表示种群，每行表示一个粒子
        # 初始化粒子的位置，单个粒子的位置由2*wn个数组成，由wn个x坐标和wn个y坐标排列而成
        X = np.random.uniform(3, 8, size=(self.size, int(self.dim / 2)))
        X = np.hstack([X, np.random.uniform(2, 10, size=(self.size, int(self.dim / 2)))])
        # 初始化种群的各个粒子的速度
        V = np.random.uniform(-1, 1, size=(self.size, self.dim))
        # 初始化粒子历史最优位置为当前位置
        pbest = X
        # ===计算每个粒子的适应度以及更新后的X[i]======
        for i in range(self.size):
            x_part, info[i, 1] = self.calc_f(X[i])  # 目标函数值
            X[i] = x_part
        # ===历史最优=====
        info[:, 0] = info[:, 1]  # 粒子的历史最优位置对应的适应度值
        # ==== 全局最优=======
        gbest_i = info[:, 0].argmin()  # 全局最优对应的粒子编号
        gbest = X[gbest_i]  # 全局最优粒子的位置
        gbest_fitness = info[gbest_i, 0]  # 全局最优位置对应的适应度值
        # === 记录迭代过程的最优适应度值====
        fitneess_value_list.append(gbest_fitness)
        # ====================================================接下来开始迭代====================================================
        for _ in tqdm(range(self.iter_num)):
            # ==更新速度==
            V = self.velocity_update(V, X, pbest=pbest, gbest=gbest)
            # ==更新位置==
            X = self.position_update(X, V)
            # ==计算每个粒子的适应度==
            for i in range(self.size):
                x_part, info[i, 1] = self.calc_f(X[i])  # 目标函数值
                X[i] = x_part
            # ===更新历史最优位置=====
            for i in range(self.size):
                pbesti = pbest[i]
                pbest_fitness = info[i, 0]  # 该粒子对应的最优适应度
                xi = X[i]
                xi_fitness = info[i, 1]  # 该粒子当前的适应度
                # ==计算更新个体历史最优==
                pbesti, pbest_fitness = self.update_pbest(pbesti, pbest_fitness, xi, xi_fitness)
                pbest[i] = pbesti
                info[i, 0] = pbest_fitness
            # ===更新全局最优=====
            pbest_fitness = info[:, 1]
            gbest, gbest_fitness = self.update_gbest(gbest, gbest_fitness, pbest, pbest_fitness, )
            # ===记录当前迭代全局之适应度====
            fitneess_value_list.append(gbest_fitness)
            # 最后绘制适应度值曲线
        best_position,best_result=self.calc_f(gbest)
        best_x = best_position[:int(self.dim / 2)]
        best_y = best_position[int(self.dim / 2):]
        for i in range(int(self.dim/2)):
            print('第{}个配送中心的位置坐标为：{}，{}'.format(i,best_x[i],best_y[i]))
        print('最优目标函数值为：{}'.format(best_result))
        # print('迭代最优结果是：{}'.format(self.calc_f(gbest)))
        # print('迭代最优变量是：{}'.format(gbest[0], gbest[1]))
        # 绘图
        plt.plot(fitneess_value_list, color='r')
        plt.title('迭代过程')
        plt.show()


if __name__ == "__main__":
    # nw 配总中心个数
    # w：物流量
    # h：运输费率
    # p：仓库坐标
    nw = 2
    p = np.array([[3, 8], [8, 2], [2, 5], [6, 10], [8, 8]])
    w = [2000, 3000, 2500, 1000, 1500]
    h = [0.05, 0.05, 0.075, 0.075, 0.075]
    # h = [1, 1, 1, 1, 1]
    xrange = [p[:, 0].min(), p[:, 0].max()]
    yrange = [p[:, 1].min(), p[:, 1].max()]

    li = Liqun(nw, p, w, h, xrange, yrange)
    li.main()
