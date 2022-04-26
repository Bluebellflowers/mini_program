# ==========导入包======================
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  ## 解决保存图像是负号'-'显示为方块的问题


class Liqun:
    def __init__(self):
        # Pso参数：
        self.w = 1  # 惯性因子，一般取 1
        self.c1 = 2  # 学习因子 1，一般取 2
        self.c2 = 2  # 学习因子 2，一般取 2
        self.dim = 2  # 维度的维度#对应 2 个参数 x,y (针对具体问题）
        self.size = 100  # 种群大小，即种群中小鸟的个数
        self.iter_num = 1000  # 算法最大迭代次数
        self.max_vel = 0.5  # 限制粒子的最大速度为 0.5
        self.min_vel = -0.5  # 限制粒子的最小速度为-0.5

    # ============= 目标函数公式：============================
    def calc_f(self, X):  # 计算个体粒子的的适应度值，也就是目标函数值，X 的维度是 size * 2
        a = 10
        x = X[0]
        y = X[1]
        return 2 * a + x ** 2 - a * np.cos(2 * np.pi * x) + y ** 2 - a * np.cos(2 * np.pi * y)

    # ===========惩罚项公式：===============================
    def calc_e1(self, X):  # 个体惩罚项,计算第一个约束的惩罚项
        e = X[0] + X[1] - 6
        return max(0, e)

    def calc_e2(self, X):  # 个体惩罚项,计算第二个约束的惩罚项
        e = 3 * X[0] - 2 * X[1] - 5
        return max(0, e)

    # ========== 由惩罚项计算Lj（权重值）：====================
    def calc_Lj(self, e1, e2):  # 根据每个粒子的约束惩罚项计算Lj权重值，e1, e2列向量，表示每个粒子的第1个第2个约束的惩罚项值
        if (e1.sum() + e2.sum()) <= 0:  # 注意防止分母为零的情况
            return 0, 0
        else:
            L1 = e1.sum() / (e1.sum() + e2.sum())
            L2 = e2.sum() / (e1.sum() + e2.sum())
        return L1, L2

    # =======粒子群速度更新公式：============================
    def velocity_update(self, V, X, pbest, gbest):
        # 根据速度更新公式更新每个粒子的速度
        # 种群 size
        #:param V: 粒子当前的速度矩阵，size*dim 的矩阵
        #:param X: 粒子当前的位置矩阵，size*dim 的矩阵
        #:param pbest: 每个粒子历史最优位置，size*dim 的矩阵
        #:param gbest: 种群历史最优位置，1*dim 的矩阵
        r1 = np.random.random((self.size, 1))
        r2 = np.random.random((self.size, 1))
        V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)  # 直接对照公式写就好了
        # 防止越界：
        V[V > self.max_vel] = self.max_vel
        V[V < self.min_vel] = self.min_vel
        return V

    # ===========粒子群位置更新公式：============================
    def position_update(self, X, V):
        # 根据公式更新粒子的位置
        #:param X: 粒子当前的位置矩阵，维度是 size*dim
        #:param V: 粒子当前的速度举着，维度是 size*dim
        X = X + V
        size = np.shape(X)[0]
        for i in range(size):
            if X[i][0] <= 1 or X[i][0] >= 2:
                X[i][0] = np.random.uniform(1, 2, 1)[0]
            if X[i][1] <= -1 or X[i][1] >= 0:
                X[i][1] = np.random.uniform(-1, 0, 1)[0]
        return X

    # =========== 更新种群函数：=============================
    def update_pbest(self, pbest, pbest_fitness, pbest_e, xi, xi_fitness, xi_e):
        # 判断是否需要更新粒子的历史最优位置
        #:param pbest: 历史最优位置
        #:param pbest_fitness: 历史最优位置对应的适应度值
        #:param pbest_e: 历史最优位置对应的约束惩罚项
        #:param xi: 当前位置
        #:param xi_fitness: 当前位置的适应度函数值
        #:param xi_e: 当前位置的约束惩罚项
        #:return:
        A = 0.0000001
        # 下面的 0.0000001 是考虑到计算机的数值精度位置，值等同于 0,不同问题设置不同的可接受值
        # 1、如果pbest和xi都没有违反约束，则取适应度（目标函数）小的：
        if pbest_e <= A and xi_e <= A:
            if pbest_fitness <= xi_fitness:
                return pbest, pbest_fitness, pbest_e
            else:
                return xi, xi_fitness, xi_e
        # 规则 2，如果当前位置违反约束而历史最优没有违反约束，则取历史最优
        if pbest_e <= A and xi_e >= A:
            return pbest, pbest_fitness, pbest_e
        # 规则 3，如果历史位置违反约束而当前位置没有违反约束，则取当前位置
        if pbest_e >= A and xi_e <= A:
            return xi, xi_fitness, xi_e
        # 4、如果两个都违反了，取适应度较小的一个：
        if pbest_fitness <= xi_fitness:
            return pbest, pbest_fitness, pbest_e
        else:
            return xi, xi_fitness, xi_e

    # ==================更新全局最优位置====================================
    def update_gbest(self, gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e):
        A = 0.0000001  # 可接受阈值，不同问题修改为不同的值
        # 先对种群，寻找约束惩罚项=0 的最优个体，如果每个个体的约束惩罚项都大于 0，就找适应度最小的个体
        pbest2 = np.concatenate([pbest, pbest_fitness.reshape(-1, 1), pbest_e.reshape(-1, 1)],
                                axis=1)  # 将几个矩阵拼接成矩阵 ，4 维矩阵（x,y,fitness,e）
        pbest2_1 = pbest2[pbest2[:, -1] <= A]  # 找出没有违反约束的个体
        if len(pbest2_1) > 0:
            pbest2_1 = pbest2_1[pbest2_1[:, 2].argsort()]  # 根据适应度值排序
        else:
            pbest2_1 = pbest2[pbest2[:, 2].argsort()]  # 如果所有个体都违反约束，直接找出适应度值最小
        # 当前迭代的最优个体
        pbesti, pbesti_fitness, pbesti_e = pbest2_1[0, :2], pbest2_1[0, 2], pbest2_1[0, 3]
        # 当前最优和全局最优比较
        # 如果两者都没有约束
        if gbest_e <= A and pbesti_e <= A:
            if gbest_fitness < pbesti_fitness:
                return gbest, gbest_fitness, gbest_e
            else:
                return pbesti, pbesti_fitness, pbesti_e
            if gbest_e <= A and pbesti_e > A:
                return gbest, gbest_fitness, gbest_e
            if gbest_e > A and pbesti_e <= A:
                return pbesti, pbesti_fitness, pbesti_e
            # 如果都违反约束，直接取适应度小的
            if gbest_fitness < pbesti_fitness:
                return gbest, gbest_fitness, gbest_e
            else:
                return pbesti, pbesti_fitness, pbesti_e

    # =============== 主函数=============================
    def main(self):
        # 初始化一个矩阵 info, 记录：
        # 0、种群每个粒子的历史最优位置对应的适应度，
        # 1、历史最优位置对应的惩罚项，
        # 2、当前适应度，
        # 3、当前目标函数值，
        # 4、约束 1 惩罚项，
        # 5、约束 2 惩罚项，
        # 6、惩罚项的和
        # 所以列的维度是 7
        info = np.zeros((self.size, 7))
        fitneess_value_list = []  # 记录每次迭代过程中的种群适应度值变化
        # 用一个 size*dim 的矩阵表示种群，每行表示一个粒子
        X = np.random.uniform(-5, 5, size=(self.size, self.dim))
        # 初始化种群的各个粒子的速度
        V = np.random.uniform(-0.5, 0.5, size=(self.size, self.dim))
        # 初始化粒子历史最优位置为当前位置
        pbest = X
        # ===计算每个粒子的适应度======
        for i in range(self.size):
            info[i, 3] = self.calc_f(X[i])  # 目标函数值
            info[i, 4] = self.calc_e1(X[i])  # 第一个约束的惩罚项
            info[i, 5] = self.calc_e2(X[i])  # 第二个约束的惩罚项
        # === 计算惩罚项的权重，及适应度值====
        L1, L2 = self.calc_Lj(info[i, 4], info[i, 5])
        info[:, 2] = info[:, 3] + L1 * info[:, 4] + L2 * info[:, 5]  # 适应度值
        info[:, 6] = L1 * info[:, 4] + L2 * info[:, 5]  # 惩罚项的加权求和
        # ===历史最优=====
        info[:, 0] = info[:, 2]  # 粒子的历史最优位置对应的适应度值
        info[:, 1] = info[:, 6]  # 粒子的历史最优位置对应的惩罚项值
        # ==== 全局最优=======
        gbest_i = info[:, 0].argmin()  # 全局最优对应的粒子编号
        gbest = X[gbest_i]  # 全局最优粒子的位置
        gbest_fitness = info[gbest_i, 0]  # 全局最优位置对应的适应度值
        gbest_e = info[gbest_i, 1]  # 全局最优位置对应的惩罚项
        # === 记录迭代过程的最优适应度值====
        fitneess_value_list.append(gbest_fitness)
        # ================接下来开始迭代==================
        for j in tqdm(range(self.iter_num)):
            # ==更新速度==
            V = self.velocity_update(V, X, pbest=pbest, gbest=gbest)
            # ==更新位置==
            X = self.position_update(X, V)
            # ==计算每个粒子的目标函数和约束惩罚项==
            for i in range(self.size):
                info[i, 3] = self.calc_f(X[i])  # 目标函数值
                info[i, 4] = self.calc_e1(X[i])  # 第一个约束的惩罚项
                info[i, 5] = self.calc_e2(X[i])  # 第二个约束的惩罚项
            # ===计算惩罚项的权重，及适应度值===
            L1, L2 = self.calc_Lj(info[i, 4], info[i, 5])
            info[:, 2] = info[:, 3] + L1 * info[:, 4] + L2 * info[:, 5]  # 适应度值
            info[:, 6] = L1 * info[:, 4] + L2 * info[:, 5]  # 惩罚项的加权求和
            # ===更新历史最优位置=====
            for i in range(self.size):
                pbesti = pbest[i]
                pbest_fitness = info[i, 0]
                pbest_e = info[i, 1]
                xi = X[i]
                xi_fitness = info[i, 2]
                xi_e = info[i, 6]
                # ==计算更新个体历史最优==
                pbesti, pbest_fitness, pbest_e = \
                    self.update_pbest(pbesti, pbest_fitness, pbest_e, xi, xi_fitness, xi_e)
                pbest[i] = pbesti
                info[i, 0] = pbest_fitness
                info[i, 1] = pbest_e
            # ===更新全局最优=====
            pbest_fitness = info[:, 2]
            pbest_e = info[:, 6]
            gbest, gbest_fitness, gbest_e = \
                self.update_gbest(gbest, gbest_fitness, gbest_e, pbest, pbest_fitness, pbest_e)
            # ===记录当前迭代全局之适应度====
            fitneess_value_list.append(gbest_fitness)
            # 最后绘制适应度值曲线
        print('迭代最优结果是：%.5f' % self.calc_f(gbest))
        print('迭代最优变量是：x=%.5f, y=%.5f' % (gbest[0], gbest[1]))
        print('迭代约束惩罚项是：', gbest_e)
        # 绘图
        plt.plot(fitneess_value_list[: 30], color='r')
        plt.title('迭代过程')
        plt.show()


if __name__ == '__main__':
    li = Liqun()
    li.main()

