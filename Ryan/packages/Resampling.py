import pandas as pd

# This function is for resampling the tape data to a new sample rate

def resample_tapes(tape_data, resample_rate):
    """
    Resample the tape data to a new sample rate
    :param tape_data: dataframe of tape data
    :param resample_rate: new sample rate，e.g. 'D' for daily,'H' for hourly, `T` for minute...
    :return: resampled data
    """

    # 1. convert datetime
    tape_data['Date'] = pd.to_datetime(tape_data['Date'])
    tape_data['Hour'] = (tape_data['Timestamp'] // 3600).astype(int)
    tape_data['Minute'] = (tape_data['Timestamp'] // 60).astype(int) % 60
    tape_data['Timestamp'] = tape_data['Timestamp'].astype(float)

    # Generate a new column for datetime as index
    tape_data['Datetime'] = pd.to_datetime(tape_data['Date']) + pd.to_timedelta(tape_data['Hour'],
                                                                                unit='h') + pd.to_timedelta(
        tape_data['Minute'], unit='m')

    tape_data.set_index('Datetime', inplace=True)

    tape_data.head()

    # save the tape data
    tape_data.to_csv('Tapes_all_time_converted.csv')

    # 2. Resample the data
    # use the resample method to resample the data,set the frequency to 'D' for daily,'H' for hourly, `T` for minute...
    resampled_tape = tape_data.resample(resample_rate).agg({
        'Price': ['mean', 'max', 'min', 'last'],
        'Volume': 'sum'
    })

    # flatten the column names
    resampled_tape.columns = ['_'.join(col).strip() for col in resampled_tape.columns.values]

    # drop the rows with missing values
    resampled_tape = resampled_tape.dropna()


    return resampled_tape