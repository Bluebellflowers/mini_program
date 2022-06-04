# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from scipy.optimize import minimize


class DSHW(object):


    def __init__(self, y, period1, period2,
                 alpha=None, beta=None, gamma=None, omega=None,
                 exponential=True, armethod=True, method=''):

        self.y = np.array(y)
        self.period1 = period1
        self.period2 = period2

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.n = len(self.y)

        self.exponential = exponential

        self.fitted = None


    def init_params_mul(self, y):
        ratio = self.period2 // self.period1

        s1 = self.seasindex(y, self.period1)
        s2 = self.seasindex(y, self.period2)
        s2 = s2 / np.tile(s1, ratio)
        s20 = s2
        x = [0] + np.diff(y[0:self.period2]).tolist()
        t = b0 = np.mean(((y[0:self.period2] - y[self.period2:(2 * self.period2)]) / self.period2) + x) / 2
        s = l0 = np.mean(y[0:(2 * self.period2)]) - (self.period2 + 0.5) * t

        return s, t, s1, s2


    def fit_mul(self):
        # 初始值
        self.s, self.t, self.s1, self.s2 = self.init_params_mul(self.y)

        # 分配空间
        yhat = np.zeros_like(self.y, dtype=float)

        # 拟合数据
        for i in range(self.n):
            yhat[i] = (self.s + self.t) * self.s1[i % self.period1] * self.s2[i % self.period2]
            snew = self.alpha * (self.y[i] / (self.s1[i % self.period1] * self.s2[i % self.period2])) + (1 - self.alpha) * (self.s + self.t)
            tnew = self.beta * (snew - self.s) + (1 - self.beta) * self.t

            s1 = self.s1.copy()
            s2 = self.s2.copy()

            self.s1[i % self.period1] = self.gamma * (self.y[i] / (snew * s2[i % self.period2])) + (1 - self.gamma) * s1[i % self.period1]
            self.s2[i % self.period2] = self.omega * (self.y[i] / (snew * s1[i % self.period1])) + (1 - self.omega) * s2[i % self.period2]

            self.s = snew
            self.t = tnew

        return yhat


    def fit(self, method='L-BFGS-B'):
        """
        method : str, default "L-BFGS-B"
        The minimizer used. Valid options are "L-BFGS-B" , "Nelder-Mead",
        "SLSQP", "BFGS", and "least_squares" (also "ls").
        """

        if self.alpha is None or self.beta is None or self.gamma is None or self.omega is None:
            self.optim(method)

        if self.exponential:
            self.fitted = self.fit_mul()
        else:
            self.fitted = self.fit_mul()

        return self.fitted


    def forecast(self, h):
        # 预测时可人为开启一个新周期，新周期的开始可能原周期的中间部位，所以要对s1、s2进行移位
        s1 = np.roll(self.s1, -self.n % self.period1)
        s2 = np.roll(self.s2, -self.n % self.period2)

        fcast = self.s + np.arange(1, h + 1) * self.t * np.tile(s1, h // self.period1 + 1)[:h] * np.tile(s2, h // self.period2 + 1)[:h]

        return fcast


    def seasindex(self, y, p):
        n = len(y)
        n2 = 2 * p

        shorty = y[:n2]
        average = np.zeros(n)

        simplema = pd.Series(shorty).rolling(p).mean()
        simplema = simplema.dropna().values

        if p % 2 == 0:  # Even order
            offset = p // 2
            centeredma = pd.Series(simplema[0:n2 - p + 1]).rolling(2).mean()
            centeredma = centeredma.dropna().values
            average[offset:offset + p] = shorty[offset:offset + p] / centeredma[:p]
            si = average[list(range(p, p + offset)) + list(range(offset, p))]

        else:  # Odd order
            offset = (p - 1) // 2
            average[offset:offset + p] = shorty[offset:offset + p] / simplema[:p]
            si = average[list(range(p, p + offset)) + list(range(offset, p))]

        return si


    def _mse(self, y, yhat):
        y = np.array(y)

        e = y - yhat
        mse = np.mean(e ** 2)
        mape = np.mean(np.abs(e) / y) * 100

        res = mse

        return res


    def optim(self, method):
        opt_alpha = self.alpha is None
        opt_beta = self.beta is None
        opt_gamma = self.gamma is None
        opt_omega = self.omega is None

        initial_values = np.array([0.1, 0.01, 0.001, 0.001])
        boundaries = np.array([(0, 1), (0, 0), (0, 1), (0, 1)])

        def optimization_criterion(params):
            alpha, beta, gamma, omega = params
            if opt_alpha and alpha >= 0 and alpha <= 1:
                self.alpha = alpha
            if opt_beta and beta >= 0 and beta <= 1:
                self.beta = beta
            if opt_gamma and gamma >= 0 and gamma <= 1:
                self.gamma = gamma
            if opt_omega and omega >= 0 and omega <= 1:
                self.omega = omega
            if self.exponential:
                fitted = self.fit_mul()
            else:
                fitted = self.fit_mul()

            # 最小二乘待优化--限制参数取值在0-1之间
            if method == 'ls' or method == 'least_squares':
                return self.y - fitted
            else:
                mse = self._mse(self.y, fitted)
                return mse

        if method == 'ls' or method == 'least_squares':
            leastsq(optimization_criterion, x0=initial_values)
        else:
            minimize(optimization_criterion, x0=initial_values, bounds=boundaries, method='L-BFGS-B')

        # minimize(optimization_criterion, x0=initial_values, bounds=boundaries, method='Nelder-Mead')
        # minimize(optimization_criterion, x0=initial_values, bounds=boundaries, method='BFGS')
        # minimize(optimization_criterion, x0=initial_values, bounds=boundaries, method='L-BFGS-B')
        # minimize(optimization_criterion, x0=initial_values, bounds=boundaries, method='SLSQP')


    def summary(self):
        res = {
            'alpha': round(self.alpha, 4),
            'beta': round(self.beta, 4),
            'gamma': round(self.gamma, 4),
            'omega': round(self.omega, 4),
            'mse': round(self._mse(self.y, self.fitted), 4)
        }
        return res
import matplotlib.pyplot as plt
plt.figure(1)
data = [1,2,3,2,3,4, 5,6,7,6,7,8, 10,11,12,11,12,13]
plt.plot(data)
plt.show()

model = DSHW(data, period1=3, period2=6, alpha=0.7, beta=0.2, gamma=0.1, omega=0.03209894)
fitted = model.fit()

print('fitted\n', fitted)
plt.figure(1)
plt.plot(fitted)
plt.show()
print('mse\n', model._mse(data, fitted))