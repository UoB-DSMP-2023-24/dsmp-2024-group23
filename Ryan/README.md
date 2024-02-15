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

### Packages
#### 1. `Feature_extraction_tapes.py`
This script has several functions to extract features from the tape data, including:
- Rolling statistics;
- Historical difference;
- RSI
- MACD
- Bollinger Bands
- Volume Weighted Average Price (VWAP)

#### 2. `Parsing_LOB.py` AND `Parsing_LOB3.0.py`
The two scripts are used to parse the LOB data and extract features. 
- `Parsing_LOB.py` is for one LOB data file.
- `Parsing_LOB3.0.py` uses some functions from `Parsing_LOB.py` and is for extracting features and combine 
all LOB data files into one file.

#### 3. `Resampling.py`
This script is used to resample the tape data and LOB data to the same time frequency.
