import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# -*- coding: UTF-8 -*-
path_origion = pd.read_csv("./车辆轨迹数据/card_hist_00000_20210910.csv", encoding='gbk')
path_origion[["card_x", "card_y"]]
xy = path_origion[["card_x", "card_y"]].values.tolist()
x = path_origion["card_x"].tolist()
y = path_origion["card_y"].tolist()
plt.title('Trajectory')
plt.plot(x, y, color='blue', label='car1')
plt.legend()  # 显示图例
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
