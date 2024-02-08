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
- `lob_data_expanded.csv`: Parsed LOB data 
- `lob_expanded_clean.csv`: Cleaned LOB data(drop Na)
- `tape1.csv`: Parsed tape data
- `combined_tape_data.csv`: Combined tape data(all dates)
- `lob_only_features.csv`: LOB data, with new features only. Details in the `Parsing_LOB.py` file