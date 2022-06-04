import matplotlib.pyplot as plt
import matplotlib.pylab as mpl
from statsmodels.tsa.api import ExponentialSmoothing
from math import ceil

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

f = open("新建文本文档2.txt",'r')
B = []
for i in f.readlines():
    B.append(int(i))
f.close()

data = B
plt.plot(data,label='第1到104周实际数据')

fit1 = ExponentialSmoothing(
    data,
    seasonal_periods=30,
    trend="add",
    seasonal="add",
    use_boxcox=True,

    initialization_method="estimated",
).fit()
plt.figure(1)
print(fit1.summary())
plt.plot(fit1.fittedvalues,label='第1到104周拟合数据')
forecast=fit1.forecast(8)
forecast1=[]
for index,data in enumerate(forecast):
    forecast1.append(ceil(data))
plt.plot([i for i in range(len(B),len(B)+8)],forecast,label='第105到112周')
print('预测为：{}'.format(forecast1))
plt.legend()
plt.title('第1到112周曲线拟合及预测')
plt.show()
print('forecast:', fit1.forecast(3))

plt.figure(2)
plt.plot(fit1.fittedvalues,label='第1到104周实际数据')
forecast=fit1.forecast(100)
plt.plot([i for i in range(len(B),len(B)+100)],forecast,label='预测之后100周的趋势变化')
plt.legend()
plt.title('第1到212周曲线拟合及预测')
plt.show()