import ast  # parse string representation of list to list
import json
import pandas as pd
# This module provides several functions to parse the LOB data and extract new features
# input: one LOB data file
# output: new csv file only containing the new features(10 columns)
# The aim of this script is to reduce the size of the original LOB data and make it easier to use for further analysis

# read file
# parse orders--> extract bid and ask lists from the Orders column
# drop NaN
# expand orders, which includes:
# - Total Bid Quantity: total quantity of all bid orders
# - Total Ask Quantity: total quantity of all ask orders
# - Max Bid Price: highest bid price
# - Min Ask Price: lowest ask price
# - Spread: difference between the lowest ask price and the highest bid price
# - Weighted Avg Bid Price: weighted average bid price
# - Weighted Avg Ask Price: weighted average ask price
# - Bid-Ask Quantity Ratio: ratio of total bid quantity to total ask quantity

# read the lob using the parse_lob_entry function and read_lines
def read_lob_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lob_data = []
        for line in lines:
            # Ensure Exch0 is treated as a string
            fixed_line = line.replace('Exch0', "'Exch0'")
            lob_data.append(ast.literal_eval(fixed_line))
    return pd.DataFrame(lob_data, columns=["Timestamp", "Exchange", "Orders"])


# Parse Orders column and add new columns to the DataFrame for the bid and ask prices and quantities
def parse_orders(row):
    # Extract the bid and ask lists from the Orders column
    bids = row['Orders'][0][1]
    asks = row['Orders'][1][1]

    bid_price = bid_quantity = ask_price = ask_quantity = None

    if bids:
        bid_price, bid_quantity = bids[0]  # extract the price and quantity of the first bid

    if asks:
        ask_price, ask_quantity = asks[0]  # extract the price and quantity of the first ask

    return pd.Series([bid_price, bid_quantity, ask_price, ask_quantity])


# Calculate additional features
def expand_orders(row):
    bid_orders = row['Orders'][0][1]
    ask_orders = row['Orders'][1][1]

    # Initialize variables
    total_bid_quantity = total_ask_quantity = total_bid_value = total_ask_value = 0
    max_bid_price = float('-inf')
    min_ask_price = float('inf')

    for price, quantity in bid_orders:
        total_bid_quantity += quantity
        total_bid_value += price * quantity
        max_bid_price = max(max_bid_price, price)

    for price, quantity in ask_orders:
        total_ask_quantity += quantity
        total_ask_value += price * quantity
        min_ask_price = min(min_ask_price, price)

    spread = min_ask_price - max_bid_price if min_ask_price != float('inf') and max_bid_price != float('-inf') else None
    weighted_avg_bid_price = total_bid_value / total_bid_quantity if total_bid_quantity > 0 else None
    weighted_avg_ask_price = total_ask_value / total_ask_quantity if total_ask_quantity > 0 else None
    bid_ask_quantity_ratio = total_bid_quantity / total_ask_quantity if total_ask_quantity > 0 else None

    return pd.Series(
        [total_bid_quantity, total_ask_quantity, max_bid_price, min_ask_price, spread, weighted_avg_bid_price,
         weighted_avg_ask_price, bid_ask_quantity_ratio])


if __name__ == "__main__":
    file_path = '/Ryan/datasets/UoB_Set01_2025-01-02LOBs.txt'  # Update the file path according to your environment
    lob_data = read_lob_file(file_path)

    # Apply the parse_orders function to each row and create new columns for bid and ask prices and quantities
    lob_data[['Bid Price', 'Bid Quantity', 'Ask Price', 'Ask Quantity']] = lob_data.apply(parse_orders, axis=1)

    # Apply the expand_orders function to each row to calculate additional features
    lob_data[['Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Price', 'Min Ask Price', 'Spread',
              'Weighted Avg Bid Price', 'Weighted Avg Ask Price', 'Bid-Ask Quantity Ratio']] = lob_data.apply(
        expand_orders, axis=1)

    selected_columns = ['Timestamp', 'Exchange', 'Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Price',
                        'Min Ask Price', 'Spread', 'Weighted Avg Bid Price', 'Weighted Avg Ask Price',
                        'Bid-Ask Quantity Ratio']

    new_df = lob_data[selected_columns].copy()

    # drop NaN
    new_df = new_df.dropna()
    print(new_df.head())

    # save the new features to a new csv file
    new_df.to_csv('C:\\Users\\yhb\\dsmp-2024-group23\\Ryan\\datasets\\lob_1-2.csv', index=False)
