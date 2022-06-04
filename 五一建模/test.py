r_cost=200
c_cost=100
c_week=5
r_week=10
c_train=10
weeks=8

import numpy as np
import pandas as pd
demand=pd.read_excel("demand.xlsx",header=None)
demand=np.array(demand)
# print(demand)

question1_week_demand=demand[0]

"""
c需求量 a
r需求量 b
c购买量 x
r购买量 y
"""

from docplex.mp.model import Model
from docplex.mp.conflict_refiner import ConflictRefiner
from docplex.mp.conflict_refiner import ConflictRefinerResult
cp = [14]
rp = [0]
weeks = question1_week_demand
for index, week in enumerate(weeks[1:]):
    total_cp = 0
    total_rp=0
    demand = week
    model = Model()
    var = model.continuous_var_list([i for i in range(0, 2)], lb=0, name=['xi', 'yi'])
    for i in range(index):
        total_cp += cp[index]
        total_rp += rp[index]
    obj = (100 + 10) * var[0] + 10 * (var[0] / 10) + 200 * var[1] + \
          5 * (50 + total_cp - 4 * demand - var[0] / 10) + \
          10 * (13 + total_rp - demand)
    model.minimize(obj)
    # 添加约束条件
    model.add_constraint(50 + total_cp+var[0] -4*weeks[index-1]  >= 4 * weeks[index])
    model.add_constraint(13 + total_rp+var[1] -weeks[index-1] >= weeks[index])
    model.add_constraint(var[0] >= 0)
    model.add_constraint(var[1] >= 0)
    sol = model.solve()
    print(sol.get_values(var))
    xi,yi=sol.get_values(var)
    cp.append(xi)
    rp.append(yi)
    if index==6:
        break
