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