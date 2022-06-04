import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


def draw(data, x, y, label1, tittle, type1='bo', xx="周数", yy="数量"):
    plt.plot(data[x], data[y], type1, label=label1)
    plt.title(tittle)
    plt.xlabel(xx)
    plt.ylabel(yy)
    plt.legend()
    plt.show()


data2 = pd.read_excel('第二问.xlsx')
# draw(data2, '周次', '购买的容器艇数量', '购买的容器艇数量', '购买的容器艇数量变化情况', '-')
# draw(data2, '周次', '总成本', '总成本', '总成本变化曲线', '-', '周数', '元')
# draw(data2, '周次', '购买的操作手数量', '购买的操作手数量', '购买的操作手数量变化情况', '-')
#
data3 = pd.read_excel('第三问.xlsx')
# draw(data3, '周次', '购买的容器艇数量', '购买的容器艇数量', '购买的容器艇数量变化情况', '-')
# draw(data3, '周次', '总成本', '总成本', '总成本变化曲线', '-', '周数', '元')
# draw(data3, '周次', '购买的操作手数量', '购买的操作手数量', '购买的操作手数量变化情况', '-')

data4 = pd.read_excel('第四问.xlsx')


# draw(data4, '周次', '购买的容器艇数量', '购买的容器艇数量', '购买的容器艇数量变化情况', '-')
# draw(data4, '周次', '总成本', '总成本', '总成本变化曲线', '-', '周数', '元')
# draw(data4, '周次', '购买的操作手数量', '购买的操作手数量', '购买的操作手数量变化情况', '-')


def draw1(data, x, y, label1, tittle, type1='bo', xx="周数", yy="数量"):
    plt.plot(data[x], data[y], type1, label=label1)
    plt.title(tittle)
    plt.xlabel(xx)
    plt.ylabel(yy)
    plt.legend()
    # plt.show()


# draw1(data3,'周次','购买的操作手数量','问题3购买的操作手数量','购买的操作手数量变化情况','-')
# draw1(data2,'周次','购买的操作手数量','问题2购买的操作手数量','购买的操作手数量变化情况','-')
# plt.show()
#
draw1(data4, '周次', '总成本', '问题4总成本', '总成本变化曲线', '-', '周数', '元')
draw1(data3, '周次', '总成本', '问题3总成本', '总成本变化曲线', '-', '周数', '元')
draw1(data2, '周次', '总成本', '问题2总成本', '总成本变化曲线', '-', '周数', '元')
plt.show()
#
# draw1(data3,'周次','购买的容器艇数量','问题3购买的容器艇数量','购买的容器艇数量变化情况','-')
# draw1(data2,'周次','购买的容器艇数量','问题2购买的容器艇数量','购买的容器艇数量变化情况','-')
# plt.show()
