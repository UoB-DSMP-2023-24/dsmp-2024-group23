
## 1. Volatility---Trading Volume

The number of trades is adjusted based on current market volatility and risk tolerance to ensure trades are in line with the investor's risk tolerance.

Volatility calculation: rolling standard deviation of log returns

Number of trades: Adjusted based on volatility, risk tolerance and maximum tradable volume

## 2. RSI
The RSI (Relative Strength Index) is a momentum oscillator used to measure the speed and magnitude of changes in asset prices. It was developed by J. Welles Wilder Jr. in 1978 and is widely used in technical analysis, particularly in the stock, foreign exchange, futures and other financial markets.The RSI is designed to determine whether market conditions are overbought or oversold, providing traders with potential buy or sell signals.

#### pls search the formula of RSI(put the formula in report)

1. **First calculate the volume of price change**:
   \[
   \Delta P = P_t - P_{t-1}
   \] 

2. **Then calculate the Average Gain and Average Loss**:
   - Average Gain:
     $$\[
     \text{Gain}_{\text{avg}} = \frac{\sum_{i=1}^{n} \max(\Delta P_i, 0)}{n}
     \]$$
   - Average loss:
     $$\[
     \text{Loss}_{\text{avg}} = \frac{\sum_{i=1}^{n} \max(-\Delta P_i, 0)}{n}
     \]$$

   Here $\(n\)$ is the size of the rolling window and $\(\Delta P_i\)$ is the price change at the $\(i\)$ point in time.

3. **Calculate Relative Strength (RS)**:
   \[
   RS = \frac{\text{Gain}_{\text{avg}}}{\text{Loss}_{\text{avg}}}
   \]

4. **Final calculation of RSI**:
   $$\[
   RSI = 100 - \left(\frac{100}{1 + RS}\right)
   \\\] $$

These formulas embody the calculation of daily gains and losses from price data, the relative strength obtained through their ratio, and ultimately the value of the RSI indicator through the relative strength. the RSI quantifies in this way the overbought or oversold condition of the market.

Range of RSI values: RSI values between 0 and 100.
- Overbought: RSI values above 70 usually indicate that the market may be overbought, which may be a signal to sell.
- Oversold: RSI values below 30 usually indicate that the market may be oversold, which may be a signal to buy.
- Neutral Zone: RSI values between 30 and 70 are usually considered neutral, with no obvious buy or sell signals.

## 3. OFI--Order Flow Imbalance

reference: [Order Flow Imbalance](https://github.com/nicolezattarin/LOB-feature-analysis)


![img.png](img.png)

OFI provides a measure of how the buying and selling power in a market changes over time, which can be used as a predictor of price movement. When OFI is positive, buyer power prevails; when OFI is negative, seller power prevails.

When OFI (Order Flow Imbalance) is positive, this indicates that the increase in the number of buy orders exceeded the increase in the number of sell orders or that there were fewer buy order cancellations than sell order cancellations during the time period under consideration. This is usually interpreted as buyer power prevailing as more buy orders enter the market or fewer buy orders are cancelled, indicating that market participants may be expecting prices to rise.

Therefore, in this case, a positive OFI may be seen as a buy signal. Traders may interpret this positive OFI value as stronger buying pressure in the market and thus anticipate that prices may rise in the future.

## 4. MA--Moving Average
`lob_data['MA'] = lob_data['price'].rolling(window=5).mean()`

The Moving Average (MA) is a common tool used in technical analysis, specifically the average of n time steps.
It can be used to help identify the direction of a trend and determine levels of support and resistance. 

## 5. MACD - Moving Average Convergence Divergence
The calculation of MACD (Moving Average Convergence Divergence) involves several steps, the mathematical expressions of which are shown below in LaTeX format:

1. **Calculate the Exponential Moving Average (EMA) for two different periods**.
   - Fast EMA (usually 12 periods are chosen).
     \[
     EMA_{12} = \frac{Price_t - EMA_{12}(previous\ day)}{12+1} + EMA_{12}(previous\ day)
     \]
   - Slow EMA (26 issues are generally chosen): \
     \[
     EMA_{26} = \frac{Price_t - EMA_{26}(previous\ day)}{26+1} + EMA_{26}(previous\ day)
     \]
   Where \( Price_t \) is the price of the current cycle.

2. **Calculate MACD line**.
   - Subtract the slow EMA from the fast EMA: \[[\[(previous\ day) \]]
     \[
     MACD = EMA_{12} - EMA_{26}
     \]

3. **Calculate Signal Line**.
   - Take a shorter period EMA for the MACD line (usually 9 periods are chosen): \[ MACD = EMA_{12} - EMA_{26} \ ]
     \[
     Signal = \frac{MACD_t - Signal(previous\ day)}{9+1} + Signal(previous\ day)
     \]

4. **Calculating Histogram**.
   - The histogram is the difference between the MACD line and the signal line:.
     \[
     Histogram = MACD - Signal
     \]

Common methods of generating buy and sell signals based on MACD are as follows:

MACD (Moving Average Convergence Divergence) is a technical analysis tool commonly used in stock trading to identify trends and changes in momentum in stock prices.MACD is calculated by using the relationship between two exponentially smoothed moving averages (EMAs) of different periods.

### How to calculate MACD

MACD calculation consists of three main parts:

1. **MACD line**: usually the 12-day EMA minus the 26-day EMA.
2. **Signal line**: the 9-day EMA of the MACD line.
3. **Histogram**: the difference between the MACD line and the signal line, usually called the MACD Histogram.

The specific steps are as follows:
1. Calculate the Fast EMA (12-day EMA).
2. Calculate the Slow EMA (26-day EMA).
3. Calculate the difference between the two (Fast EMA - Slow EMA) to get the MACD line.
4. The MACD line is averaged again (usually with a 9-day EMA) to get the signal line.
5. The difference between the MACD line and the signal line is the MACD histogram.

### How to use MACD as a buy/sell signal for trading

1. **Crossing Signal**:
   - **Golden Cross**: When the MACD line crosses the signal line from below upwards, it indicates a possible buy signal.
   - **Dead Cross**: When the MACD line crosses the signal line from above downwards, it indicates a possible sell signal.

2. **Divergence**:
   - When price makes a new high and the MACD fails to make a new high, it may indicate a decline (bearish divergence).
   - When price makes a new low and the MACD fails to make a new low, it may indicate a rally (bullish divergence).

3. **Zero-axis crossover**:
   - When the MACD line crosses the zero-axis from below, it indicates that the market may be turning to an uptrend and is a signal to buy.
   - When the MACD line crosses the zero-axis from up to down, it shows that the market may be turning into a downtrend and is a signal to sell.

4. **Histogram**:
    - Changes in the MACD histogram can also be used as a reference for buy and sell signals, e.g. a change in the histogram from negative to positive may be a buy signal and from positive to negative may be a sell signal.

### Precautions for use

- MACD is more suitable for trending markets, if the market is in a range bound oscillator (e.g. LOB data), the signals from MACD may be more misleading.
- Always use MACD in conjunction with other indicators and market analyses, avoiding single reliance on any one technical indicator.
- Understand the characteristics of the assets you trade and the market behaviour and adjust the MACD parameters to suit different market conditions and trading styles.

When using MACD, you can analyse historical price data to verify its validity in specific markets and assets, so that you can formulate trading strategies more precisely.

When using the MACD, the following should be noted:

- **Lag**: All EMA-based indicators have a certain lag because they are calculated based on past price data.
- **False Alarms**: Using the MACD on its own may produce false alarms, especially in sideways or less volatile market conditions.
- **Market conditions**: MACD works better in markets with a clear trend.
- **Use in combination**: It is recommended to use MACD together with other indicators or market analysis methods to increase the accuracy of the signals.

Therefore, in practice, MACD signals should be used in conjunction with broader market analysis and possibly other technical indicators such as RSI, Bollinger Bands or support/resistance levels to make more comprehensive and prudent trading decisions.


## Normalised Profit
Normalised profit is calculated as the profit divided by the number of trades.

1. **Average Effectiveness Measurement**: Normalised Profit provides a clear metric for directly comparing the efficiency of different strategies by averaging the profit per trade, regardless of trade frequency.

2. **Elimination of the interference of trading frequency**: This method naturally eliminates the influence of the number of trades. A high-frequency trading strategy may have a seemingly high total profit, but the profit per trade may not be as good, while a low-frequency trading strategy may have a more significant profit per trade, despite the low number of trades. Normalised profits make these two scenarios comparable.

3. **Strategy Evaluation and Adjustment**: By analysing the normalised daily profits of different strategies, it is possible to more accurately assess which strategies are performing better on an average per-trade basis and to adjust the strategies based on this information to improve overall performance.

Therefore, even on a single stock dataset, normalised profit as a metric can help you better understand the performance of different trading strategies and thus make more rational strategy choices and adjustments. This approach provides a useful tool for trading strategy evaluation and is particularly suitable for situations where you want to make fair comparisons between different strategies.


# Experiment Results
## EXP1 Only OFI (+3, -3)
buy: OFI > 3
sell: OFI < -3

Total Profit:  86638
Total Normalised daily profits 21.61
Number of buys:  25517
Number of sells:  28282 

## EXP2 Only MACD (12, 26, 9)
buy:MACD line crosses signal line upwards
sell:MACD line crosses signal line downwards

Total Profit:  915604
Total Normalised daily profits 29.07
Number of buys:  14718
Number of sells:  16783

## EXP3 Only RSI (30, 70)
buy: RSI falls below the oversold area (RSI < 30) and rises above 30 again.
sell: RSI rises above the overbought area (RSI > 70) and falls below 70 again.

Total Profit:  149703
Total Normalised daily profits 44.38
Number of buys:  1560
Number of sells:  1813

## EXP4 Only LSTM (with 3 days MA of True Price)
buy: next_predict >= moving_average_3
sell: next_predict <= moving_average_3

Total Profit:  1743144
Total Normalised daily profits 61.98
Number of buys:  13429
Number of sells:  14696

## EXP5 RSI (30, 70) + MACD (12, 26, 9)
buy: RSI falls below the oversold area (RSI < 30) and rises above 30 again && MACD line crosses signal line upwards
sell: RSI rises above the overbought area (RSI > 70) and falls below 70 again && MACD line crosses signal line downwards

Total Profit:  1050810
Total Normalised daily profits 55.1
Number of buys:  9349
Number of sells:  9721

## EXP6 LSTM (with 3 days MA of True Price) + RSI (30, 70)
buy: next_predict >= moving_average_3 && RSI falls below the oversold area (RSI < 30) and rises above 30 again
sell: next_predict <= moving_average_3 && RSI rises above the overbought area (RSI > 70) and falls below 70 again

Total Profit:  36088
Total Normalised daily profits 75.34
Number of buys:  217
Number of sells:  262

## EXP7 LSTM (with 3 days MA of True Price) + OFI (+3, -3)
buy: next_predict >= moving_average_3 && OFI > 3
sell: next_predict <= moving_average_3 && OFI < -3

Total Profit:  284671
Total Normalised daily profits 71.94
Number of buys:  2199
Number of sells:  2397

## EXP8 LSTM (with 3 days MA of True Price) + MACD (12, 26, 9)
buy: next_predict >= moving_average_3 && MACD line crosses signal line upwards
sell: next_predict <= moving_average_3 && MACD line crosses signal line downwards

Total Profit:  96499
Total Normalised daily profits 64.55
Number of buys:  869
Number of sells:  900

## EXP9 LSTM Multi steps
Buy strategy: when the model's predictions for the next 5 steps are all incremental
Sell strategy: when the model's predictions for the next 5 steps are all decremental

![simulation_boxplot.png](..%2Ffigures%2Fsimulation_boxplot.png)