# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:34:32 2024

@author: keert
"""


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

#read the process LOB.csv   
df = pd.read_csv('C:/Users/keert/Downloads/processed_lob_data/processed_lob_data.csv') 

df_tape = pd.read_csv('C:/Users/keert/Downloads/Tapes_all.csv')

pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(df.head())

df.tail()

subset_df = df.iloc[:1000]

# Line Plot of Weighted Avg Bid Price and Weighted Avg Ask Price over Time
plt.figure(figsize=(10, 5))
plt.plot(subset_df['Timestamp'], subset_df['Weighted Avg Bid Price'], label='Weighted Avg Bid Price')
plt.plot(subset_df['Timestamp'], subset_df['Weighted Avg Ask Price'], label='Weighted Avg Ask Price')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Weighted Avg Bid and Ask Price over Time')
plt.legend()
plt.show()

# Histogram of Spread
plt.figure(figsize=(8, 6))
plt.hist(subset_df['Spread'], bins=20)
plt.xlabel('Spread')
plt.ylabel('Frequency')
plt.title('Histogram of Spread')
plt.show()

# Scatter Plot of Total Bid Quantity vs. Total Ask Quantity
plt.figure(figsize=(8, 6))
plt.scatter(subset_df['Total Bid Quantity'], subset_df['Total Ask Quantity'], alpha=0.5)
plt.xlabel('Total Bid Quantity')
plt.ylabel('Total Ask Quantity')
plt.title('Scatter Plot of Total Bid Quantity vs. Total Ask Quantity')
plt.show()


# Scatter Plot of Max Bid Price vs. Min Ask Price
plt.figure(figsize=(8, 6))
plt.scatter(subset_df['Max Bid Price'], subset_df['Min Ask Price'], alpha=0.5)
plt.xlabel('Max Bid Price')
plt.ylabel('Min Ask Price')
plt.title('Scatter Plot of Max Bid Price vs. Min Ask Price')
plt.show()

# Box Plot of Spread
plt.figure(figsize=(8, 6))
plt.boxplot(subset_df['Spread'])
plt.xlabel('Spread')
plt.title('Box Plot of Spread')
plt.show()

tape_data=df_tape.iloc[:1000]

# Plot price trend
plt.figure(figsize=(12, 6))
plt.plot(tape_data['Timestamp'], tape_data['Price'], label='Trade Price', color='blue')
plt.plot(subset_df['Timestamp'], subset_df['Weighted Avg Bid Price'], label='Weighted Avg Bid Price', linestyle='--', color='green')
plt.plot(subset_df['Timestamp'], subset_df['Weighted Avg Ask Price'], label='Weighted Avg Ask Price', linestyle='--', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Price')
plt.title('Price Trend Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot volume trend
plt.figure(figsize=(12, 6))
plt.plot(tape_data['Timestamp'], tape_data['Volume'].cumsum(), label='Trade Volume', color='blue')
plt.plot(subset_df['Timestamp'], subset_df['Total Bid Quantity'].cumsum(), label='Total Bid Quantity', linestyle='--', color='green')
plt.plot(subset_df['Timestamp'], subset_df['Total Ask Quantity'].cumsum(), label='Total Ask Quantity', linestyle='--', color='red')
plt.xlabel('Timestamp')
plt.ylabel('Volume')
plt.title('Volume Trend Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot spread over time
plt.figure(figsize=(12, 6))
plt.plot(subset_df['Timestamp'], subset_df['Spread'], label='Spread', color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Spread')
plt.title('Spread Over Time')
plt.legend()
plt.grid(True)
plt.show()