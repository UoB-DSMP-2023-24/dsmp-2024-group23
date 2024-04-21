#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


# # 1. Data Preprocessing

# In[4]:


lob_data= pd.read_csv("C:/Users/keert/OneDrive/Desktop/Mini Project/resampled_lob_secs2.csv")
lob_data['Datetime'] = pd.to_datetime(lob_data['Datetime'])

lob_data['price']=(lob_data['Max Bid Price']*lob_data['Max Bid Quantity']+lob_data['Min Ask Price']*lob_data['Min Ask Quantity'])/(lob_data['Max Bid Quantity']+lob_data['Min Ask Quantity'])
# lob_data


# In[5]:


# ofi
def calculate_ofi(df):
    df['delta_bid'] = df['Max Bid Price'].diff().fillna(0)
    df['delta_ask'] = df['Min Ask Price'].diff().fillna(0)
    
        # 计算 ΔW^m(t_n)
    df['delta_w'] = df.apply(
        lambda row: row['Max Bid Quantity'] if row['delta_bid'] > 0 else
                   (-row['Max Bid Quantity'] if row['delta_bid'] < 0 else
                    row['Max Bid Quantity'] - df.loc[df.index[df.index.get_loc(row.name)-1], 'Max Bid Quantity']
                    if row.name > 0 else 0),
        axis=1)
    
    # 计算 ΔV^m(t_n)
    df['delta_v'] = df.apply(
        lambda row: -row['Min Ask Quantity'] if row['delta_ask'] > 0 else
                   (row['Min Ask Quantity'] if row['delta_ask'] < 0 else
                    row['Min Ask Quantity'] - df.loc[df.index[df.index.get_loc(row.name)-1], 'Min Ask Quantity']
                    if row.name > 0 else 0),
        axis=1)
    
    # 计算 OFI
    df['ofi'] = df['delta_w'] + df['delta_v']
    
    return df['ofi']

calculate_ofi(lob_data)
lob_data=lob_data[['Datetime','Min Ask Price','Max Bid Price','price','Total Bid Quantity','Total Ask Quantity','Min Ask Quantity','Max Bid Quantity','ofi']]
# drop the first row (ofi is 0)
lob_data = lob_data.iloc[1:]


# In[6]:


def calculate_rsi(data, window):
    """
    计算给定数据的相对强弱指数（RSI）。
    
    :param data: 包含价格数据的Pandas Series。
    :param window: 用于计算RSI的窗口大小，默认为14。
    :return: 包含RSI值的Pandas Series。
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

lob_data['RSI'] = calculate_rsi(lob_data['price'], window=5) # calculate RSI
lob_data['MA'] = lob_data['price'].rolling(window=5).mean() # Cclculate moving avg
lob_data['Momentum'] = lob_data['price'] - lob_data['price'].shift(5) # Calculate momentum
# lob_data.head()


# In[7]:


# This is only for trading volume, not for modeling
# # 设置一个时间窗口
rolling_window = 5  # 例如，我们使用过去5个时间点的数据来计算波动率

# 计算对数收益率 shift(1)是为了计算相对于前一分钟的收益率
lob_data['Log Return Max Bid'] = np.log(lob_data['Max Bid Price'] / lob_data['Max Bid Price'].shift(1))
lob_data['Log Return Min Ask'] = np.log(lob_data['Min Ask Price'] / lob_data['Min Ask Price'].shift(1))

# 计算滚动标准差作为波动率的度量
lob_data['Volatility Max Bid'] = lob_data['Log Return Max Bid'].rolling(window=rolling_window).std()
lob_data['Volatility Min Ask'] = lob_data['Log Return Min Ask'].rolling(window=rolling_window).std()

# 由于滚动计算会产生缺失值，我们通常会删除这些值
# lob_data.dropna(inplace=True)

# Trading volume
def adjust_trade_quantity(volatility, max_tradeable_quantity, base_quantity=1, risk_tolerance=0.5, scaler_fator=10):
    """
    根据波动率和最大可交易量调整交易量。
    volatility: 当前波动率
    max_tradeable_quantity: 该时间点的最大可交易量（对于买入操作，是Min Ask Quantity；对于卖出操作，是Max Bid Quantity）
    base_quantity: 基础交易量
    risk_tolerance: 风险容忍度，取值范围为[0, 1]，数值越小表风险承受越大，交易量越大
    """
    # 基于波动率调整的交易量
    adjusted_quantity = base_quantity / (volatility*risk_tolerance*scaler_fator)
    adjusted_quantity = max(1, round(adjusted_quantity))  # 确保至少交易1单位，并且是整数

    # 确保交易量不超过最大可交易量 make sure the trade quantity is no more than the max tradeable quantity
    final_trade_quantity = min(adjusted_quantity, max_tradeable_quantity)

    return final_trade_quantity


# In[8]:


# split data
lob_data.columns


# In[9]:


# drop na
lob_data.dropna(inplace=True)

# split data
split_index = int(0.8 * len(lob_data))
features = ['Total Bid Quantity', 'Total Ask Quantity', 'Min Ask Quantity', 'Max Bid Quantity', 'ofi', 'RSI', 'MA', 'Momentum']
target = 'price'
X_train, X_test = lob_data[features][:split_index], lob_data[features][split_index:]
y_train, y_test = lob_data[target][:split_index], lob_data[target][split_index:]


# In[10]:


lob_data


# 
# # 2. XGBoost Model
# 使用arima模型预测price发现效果并不是很好，因为arima基于线性模型，而price的变化是非线性的，因此我们需要使用更加复杂的模型来预测price。
# 
# Sklearn提供了TimeseriesSplit函数，可以用于时间序列数据的交叉验证。我们可以使用这个函数来划分训练集和测试集，然后使用XGBoost模型进行预测。

# In[11]:


# 使用XGBoost模型进行预测
# # 标准化特征,只对非时间特征进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[12]:


# 使用TimeSeriesSplit进行交叉验证，并使用grid search寻找最佳参数
tscv = TimeSeriesSplit(n_splits=5)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001]
}

# 使用GridSearchCV进行交叉验证
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=tscv, scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数
grid_search.best_params_ # {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}


# In[13]:


# 使用最佳参数训练模型
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42,n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

# 预测测试集
y_pred_xgb = xgb_model.predict(X_test_scaled)

# 计算测试集的MSE
mse = np.mean((y_pred_xgb - y_test) ** 2)
# 计算R^2
r2 = xgb_model.score(X_test, y_test)
mse, r2


# In[14]:


y_test_value=y_test.values
# 对比预测值和真实值,
plt.figure(figsize=(12, 6))
plt.plot(y_test_value[1000:1100], label='True Price')
plt.plot(y_pred_xgb[1000:1100] ,label='Predicted Price')
plt.title('Comparison of True and Predicted Price for XGBoost Model')
plt.xlabel('Time Min')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[37]:


# Assuming lob_data is your main DataFrame and has the same index as X_test
test_data = lob_data.loc[X_test.index]


# In[38]:


# Assign predictions to the test set DataFrame
test_data['predicted_price'] = y_pred_xgb

# Verify that there are no NaN values after assignment
print("NaNs in 'predicted_price' after assignment:", test_data['predicted_price'].isna().sum())


# In[40]:


# Generate trade signals based on the predicted prices
test_data['trade_signal'] = test_data.apply(
    lambda row: 1 if row['predicted_price'] > row['price'] else -1, axis=1
)


# In[41]:


def trading_simulator(data, initial_capital=10000):
    capital = initial_capital
    position = 0
    current_hold_value = 0
    
    for _, row in data.iterrows():
        # Buy signal
        if row['trade_signal'] == 1 and capital >= row['price']:
            num_shares_to_buy = capital // row['price']
            capital -= num_shares_to_buy * row['price']
            position += num_shares_to_buy
        # Sell signal
        elif row['trade_signal'] == -1 and position > 0:
            capital += position * row['price']
            position = 0
        # Update current holding value
        current_hold_value = position * row['price']
    
    # Final portfolio value is cash plus the current holding value
    final_portfolio_value = capital + current_hold_value
    profit = final_portfolio_value - initial_capital
    return profit, capital, position


# In[52]:


# Print summary statistics to check for any abnormalities
print(test_data[['trade_signal', 'price', 'predicted_price']].describe())

# Check for NaN counts explicitly
print(test_data[['trade_signal', 'price', 'predicted_price']].isna().sum())


# In[53]:


def trading_simulator(data, initial_capital=10000):
    capital = initial_capital
    position = 0
    current_hold_value = 0
    
    for index, row in data.iterrows():
        # Debugging outputs
        print(f"Row {index}: Signal={row['trade_signal']}, Price={row['price']}, Predicted={row['predicted_price']}")

        if np.isnan(row['trade_signal']) or np.isnan(row['price']) or np.isnan(row['predicted_price']):
            print("Skipping due to NaN")
            continue

        if row['trade_signal'] == 1:  # Buy signal
            if capital >= row['price']:
                num_shares_to_buy = capital // row['price']
                capital -= num_shares_to_buy * row['price']
                position += num_shares_to_buy
            print(f"Bought {num_shares_to_buy} shares, Capital now: {capital}, Position: {position}")

        elif row['trade_signal'] == -1:  # Sell signal
            if position > 0:
                capital += position * row['price']
                position = 0
            print(f"Sold all shares, Capital now: {capital}, Position: {position}")

        current_hold_value = position * row['price']
        print(f"Current Holding Value: {current_hold_value}")

    final_portfolio_value = capital + current_hold_value
    profit = final_portfolio_value - initial_capital
    
    return capital, position, profit

# Run the simulation again with debugging
capital, position, profit = trading_simulator(test_data, initial_capital=10000)


# In[54]:


def trading_simulator(data, initial_capital=10000):
    capital = initial_capital
    position = 0
    current_hold_value = 0
    
    for index, row in data.iterrows():
        # Debugging outputs
        print(f"Row {index}: Signal={row['trade_signal']}, Price={row['price']}, Predicted={row['predicted_price']}")

        if np.isnan(row['trade_signal']) or np.isnan(row['price']) or np.isnan(row['predicted_price']):
            print("Skipping due to NaN")
            continue

        if row['trade_signal'] == 1:  # Buy signal
            if capital >= row['price']:
                num_shares_to_buy = capital // row['price']
                capital -= num_shares_to_buy * row['price']
                position += num_shares_to_buy
            print(f"Bought {num_shares_to_buy} shares, Capital now: {capital}, Position: {position}")

        elif row['trade_signal'] == -1:  # Sell signal
            if position > 0:
                capital += position * row['price']
                position = 0
            print(f"Sold all shares, Capital now: {capital}, Position: {position}")

        current_hold_value = position * row['price']
        print(f"Current Holding Value: {current_hold_value}")

    final_portfolio_value = capital + current_hold_value
    profit = final_portfolio_value - initial_capital
    
    return capital, position, profit

# Run the simulation again with debugging
capital, position, profit = trading_simulator(test_data, initial_capital=10000)


# In[50]:


# Run the simulation on the test data
profit, ending_capital, final_position = trading_simulator(lob_data.loc[y_test.index, :])

print(f"Ending capital after trading: ${ending_capital:.2f}")
print(f"Final position (number of shares held): {final_position}")
print(f"Profit from trading: ${profit:.2f}")

