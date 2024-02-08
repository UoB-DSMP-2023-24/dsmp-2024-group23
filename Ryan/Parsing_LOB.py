import ast  # parse string representation of list to list
import pandas as pd

# This script is used to parse the Orders column in the LOB data and calculate the following features:
# - Total Bid Quantity: total quantity of all bid orders
# - Total Ask Quantity: total quantity of all ask orders
# - Max Bid Price: highest bid price
# - Min Ask Price: lowest ask price
# - Spread: difference between the lowest ask price and the highest bid price
# - Weighted Avg Bid Price: weighted average bid price
# - Weighted Avg Ask Price: weighted average ask price
# - Bid-Ask Quantity Ratio: ratio of total bid quantity to total ask quantity
#
# Then save only the new features to a new csv file,
# I will try to apply the same implementation to all the LOB data


# read the cleaned LOB data
lob_cleaned_path = './datasets/lob_expanded_clean.csv'
lob_cleaned = pd.read_csv(lob_cleaned_path)

# Function to parse the orders and calculate the features
def parse_orders(row):
    orders = ast.literal_eval(row['Orders'])
    bid_orders = orders[0][1]
    ask_orders = orders[1][1]

    # Initialize variables
    total_bid_quantity, total_ask_quantity = 0, 0 # total bid and ask quantities
    total_bid_value, total_ask_value = 0, 0 # total bid and ask values,values= price * quantity
    max_bid_price, min_ask_price = float('-inf'), float('inf') # max bid and min ask prices

    for price, quantity in bid_orders:
        total_bid_quantity += quantity
        total_bid_value += price * quantity
        max_bid_price = max(max_bid_price, price)

    for price, quantity in ask_orders:
        total_ask_quantity += quantity
        total_ask_value += price * quantity
        min_ask_price = min(min_ask_price, price)

    # Calculate additional features
    spread = min_ask_price - max_bid_price  # spread: difference between the lowest ask price and the highest bid price
    weighted_avg_bid_price = total_bid_value / total_bid_quantity if total_bid_quantity > 0 else 0
    weighted_avg_ask_price = total_ask_value / total_ask_quantity if total_ask_quantity > 0 else 0
    bid_ask_quantity_ratio = total_bid_quantity / total_ask_quantity if total_ask_quantity > 0 else 0

    return pd.Series([total_bid_quantity, total_ask_quantity, max_bid_price, min_ask_price, spread, weighted_avg_bid_price, weighted_avg_ask_price, bid_ask_quantity_ratio])

# add the new features to the dataframe
lob_cleaned[['Total Bid Quantity', 'Total Ask Quantity', 'Max Bid Price', 'Min Ask Price', 'Spread', 'Weighted Avg Bid Price', 'Weighted Avg Ask Price', 'Bid-Ask Quantity Ratio']] = lob_cleaned.apply(parse_orders, axis=1)

print(lob_cleaned.head())

# save the new features to a new csv file
selected_columns = ['Timestamp', 'Exchange', 'Total Bid Quantity', 'Total Ask Quantity',
                    'Max Bid Price', 'Min Ask Price', 'Spread',
                    'Weighted Avg Bid Price', 'Weighted Avg Ask Price',
                    'Bid-Ask Quantity Ratio']

new_df = lob_cleaned[selected_columns].copy()
print(new_df.head())

new_df.to_csv('lob_only_features.csv', index=False)