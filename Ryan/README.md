# Explainaion

I clean the tape data and the LOB data for one date as an example, which includes:

- Tape data: 
    - add column names
    - `Timestamp`: convert to datetime
    - check missing values and outliers
    - plot the price and volume distribution
- LOB data:
    - parse the txt file by line
    - add column names
    - split the `Order` column into `Bid Price`, `Bid Quantity`, `Ask Price`, and `Ask Quantity`
    - check missing values and outliers

### Datasets
#### Single Date
- `lob_data_expanded.csv`: Parsed LOB data 
- `lob_expanded_clean.csv`: Cleaned LOB data(drop Na)
- `tape1.csv`: Parsed tape data
- `combined_tape_data.csv`: Combined tape data(all dates)
- `lob_only_features.csv`: LOB data, with new features only. Details in the `Parsing_LOB.py` file

#### Combined Data
- `Tapes_all.csv`: Combined tape data, with date information.
- `LOB_all.csv`: Combined LOB data, drop the original columns and add some new features:
  - Total Bid Quantity: total quantity of all bid orders
  - Total Ask Quantity: total quantity of all ask orders
  - Max Bid Price: highest bid price
  - Min Ask Price: lowest ask price
  - Spread: difference between the lowest ask price and the highest bid price
  - Weighted Avg Bid Price: weighted average bid price
  - Weighted Avg Ask Price: weighted average ask price
  - Bid-Ask Quantity Ratio: ratio of total bid quantity to total ask quantity

## Feature Engineering
In the notebook `Feature_Engineering.ipynb`, I created some new features for the combined Tapes `Tapes_all.csv` and
used one LOB as an example. 

### Tapes
Using the [Feature_extraction.py](packages%2FFeature_extraction.py) package for extracting features from the Tapes data

### LOB
Using [Parsing_LOB.py](packages%2FParsing_LOB.py) and [Parsing_LOB3.0.py](packages%2FParsing_LOB3.0.py)

But you can just use the combined LOB data `processed_lob_data.csv` as the results are all there.


If you want to use the feature engineering, you can check the notebook
and use the function `rolling_stats_diff` to create new features for the Tape data.

### Visualization
Visualization on tape and LOB data.
For tape data, I used tableau to do the time-series visualization on tape data. Basicly shows the trading price of the stock from 2025-01-02 to 2025-07-01. 
