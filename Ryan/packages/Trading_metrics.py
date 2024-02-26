# This file contains the functions to calculate the trading metrics.
import numpy as np


# This function calculates the Sharpe ratio.
# Sharpe ratio is a measure for calculating risk-adjusted return,
# it is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """
    计算夏普比率。

    :param daily_returns: 日收益率的数组。
    :param risk_free_rate: 无风险回报率，默认为0。
    :return: 夏普比率。
    """
    # 调整日收益率以反映无风险回报率
    adjusted_returns = daily_returns - risk_free_rate

    # 计算日收益率的平均值和标准差
    mean_return = np.mean(adjusted_returns)
    std_return = np.std(adjusted_returns)

    # 计算夏普比率，防止除以0的情况
    if std_return > 0:
        sharpe_ratio = mean_return / std_return
    else:
        sharpe_ratio = 0

    return sharpe_ratio

# # 使用封装的函数计算夏普比率
# sharpe_ratio_rsi_optimized = calculate_sharpe_ratio(daily_returns_rsi)
# sharpe_ratio_rsi_optimized
