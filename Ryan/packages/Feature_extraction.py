import pandas as pd

# This is for extracting features from combined Tape data.

# 1. Rolling statistics and historical difference
def calculat4e_rolling_stats_diff(data, column_name, time_windows):
    """
    计算给定列的滚动统计量和历史差分，并只返回这些计算结果。

    :param
    data: pd.DataFrame, 包含时间序列数据的DataFrame，索引应为DatetimeIndex。
    column_name: str, 要计算滚动统计量的列名。
    time_windows: list of str,
    # SET time window sizes
    # T = minute, H = hour, D = day.
    # eg:window_sizes = ['1T', '5T', '15T', '30T', '1H', '1D']

    :return
    pd.DataFrame, 包含给定列的滚动统计量和历史差分的DataFrame。
    """
    # initialize a empty dataframe to store the result
    result_df = pd.DataFrame(index=data.index)

    # rolling statistics
    for window in time_windows:
        result_df[f'{column_name}_RollingMean_{window}'] = data[column_name].rolling(window=window,
                                                                                     min_periods=1).mean()
        result_df[f'{column_name}_RollingStd_{window}'] = data[column_name].rolling(window=window, min_periods=1).std()
        result_df[f'{column_name}_RollingMax_{window}'] = data[column_name].rolling(window=window, min_periods=1).max()
        result_df[f'{column_name}_RollingMin_{window}'] = data[column_name].rolling(window=window, min_periods=1).min()

    # historical diff
    result_df[f'{column_name}_Diff'] = data[column_name].diff()

    return result_df

# 2. RSI
def calculate_RSI(data, column='Price', time_window='14D'):
    """
    计算给定列的相对强弱指数（RSI）。
    :param data: dataframe
    :param column: price column
    :param time_window: set time type
    :return: RSI
    """

    delta = data[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=time_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=time_window, min_periods=1).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

# 3. Bollinger Bands
def calculate_bollinger_bands(data, column='Price', time_window='20D'):
    """
    计算给定列的布林带。
    :param data: dataframe
    :param column: price column
    :param time_window: set time type
    :return: upper_band, lower_band
    """
    rolling_mean = data[column].rolling(window=time_window, min_periods=1).mean()
    rolling_std = data[column].rolling(window=time_window, min_periods=1).std()

    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)

    return rolling_mean, upper_band, lower_band

# 4. VWAP : Volume Weighted Average Price
def calculate_VWAP(data):
    """
    计算给定列的成交量加权平均价格（VWAP）。
    :param data: dataframe
    :return: VWAP
    """
    vwap = (data['Volume'] * data['Price']).cumsum() / data['Volume'].cumsum()
    return vwap


# 5. MACD
def calculate_MACD(data, column='Price', fast_period='12D', slow_period='26D', signal_period='9D'):
    # 计算快速EMA和慢速EMA
    fast_ema = data[column].ewm(span=fast_period, adjust=False, min_periods=1).mean()
    slow_ema = data[column].ewm(span=slow_period, adjust=False, min_periods=1).mean()

    # 计算MACD线
    macd_line = fast_ema - slow_ema

    # 计算信号线
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=1).mean()

    # 计算直方图
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram



