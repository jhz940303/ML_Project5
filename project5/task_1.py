import numpy as np
import pandas as pd

# Read the individual CSV files
df_DJI = pd.read_csv('Processed_DJI.csv')
df_NASDAQ = pd.read_csv('Processed_NASDAQ.csv')
df_NYSE = pd.read_csv('Processed_NYSE.csv')
df_RUSSELL = pd.read_csv('Processed_RUSSELL.csv')
df_SP = pd.read_csv('Processed_S&P.csv')

# Fill missing values with zeros
df_DJI = df_DJI.fillna(0)
df_NASDAQ = df_NASDAQ.fillna(0)
df_NYSE = df_NYSE.fillna(0)
df_RUSSELL = df_RUSSELL.fillna(0)
df_SP = df_SP.fillna(0)

# get Close column and Date column
df_Date = df_DJI["Date"]
df_DJI_Close = df_DJI["Close"]
df_NASDAQ_Close = df_NASDAQ["Close"]
df_NYSE_Close = df_NYSE["Close"]
df_RUSSELL_Close = df_RUSSELL["Close"]
df_SP_Close = df_SP["Close"]

# drop Close and Date column
df_DJI = df_DJI.drop(["Close", "Date", "Name"], axis=1)
df_NASDAQ = df_NASDAQ.drop(["Close", "Date", "Name"], axis=1)
df_NYSE = df_NYSE.drop(["Close", "Date", "Name"], axis=1)
df_RUSSELL = df_RUSSELL.drop(["Close", "Date", "Name"], axis=1)
df_SP = df_SP.drop(["Close", "Date", "Name"], axis=1)

# Combine the dataframes horizontally (by columns)
combined_df = pd.concat([df_Date, df_DJI, df_NASDAQ, df_NYSE, df_RUSSELL, df_SP], axis=1)

# add Close columns to the end of the new dataframe
combined_df["DJI_Close"] = df_SP_Close
combined_df["NASDAQ_Close"] = df_NASDAQ_Close
combined_df["NYSE_Close"] = df_NYSE_Close
combined_df["RUSSELL_Close"] = df_RUSSELL_Close
combined_df["SP_Close"] = df_SP_Close

# find direction for each Close columns
# Create a new column 'Direction' to store the 'up' or 'down' values
combined_df['DJI_Close_Direction'] = ''
combined_df['NASDAQ_Close_Direction'] = ''
combined_df['NYSE_Close_Direction'] = ''
combined_df['RUSSELL_Close_Direction'] = ''
combined_df['SP_Close_Direction'] = ''

# Iterate through the rows and compare the 'Close' value with the previous row's value
for i in range(1, len(combined_df)):
    # DJI
    if combined_df.loc[i, 'DJI_Close'] > combined_df.loc[i - 1, 'DJI_Close']:
        combined_df.loc[i, 'DJI_Close_Direction'] = 'up'
    else:
        combined_df.loc[i, 'DJI_Close_Direction'] = 'down'

    # NASDAQ
    if combined_df.loc[i, 'NASDAQ_Close'] > combined_df.loc[i - 1, 'NASDAQ_Close']:
        combined_df.loc[i, 'NASDAQ_Close_Direction'] = 'up'
    else:
        combined_df.loc[i, 'NASDAQ_Close_Direction'] = 'down'

    # NYSE
    if combined_df.loc[i, 'NYSE_Close'] > combined_df.loc[i - 1, 'NYSE_Close']:
        combined_df.loc[i, 'NYSE_Close_Direction'] = 'up'
    else:
        combined_df.loc[i, 'NYSE_Close_Direction'] = 'down'

    # RUSSELL
    if combined_df.loc[i, 'RUSSELL_Close'] > combined_df.loc[i - 1, 'RUSSELL_Close']:
        combined_df.loc[i, 'RUSSELL_Close_Direction'] = 'up'
    else:
        combined_df.loc[i, 'RUSSELL_Close_Direction'] = 'down'

    if combined_df.loc[i, 'SP_Close'] > combined_df.loc[i - 1, 'SP_Close']:
        combined_df.loc[i, 'SP_Close_Direction'] = 'up'
    else:
        combined_df.loc[i, 'SP_Close_Direction'] = 'down'



# Set the first row's 'Direction' to NaN, since there's no previous row to compare with
combined_df.loc[0, 'DJI_Close_Direction'] = "flat"
combined_df.loc[0, 'NASDAQ_Close_Direction'] = "flat"
combined_df.loc[0, 'NYSE_Close_Direction'] = "flat"
combined_df.loc[0, 'RUSSELL_Close_Direction'] = "flat"
combined_df.loc[0, 'SP_Close_Direction'] = "flat"



# drop Close columns
combined_df = combined_df.drop(['DJI_Close', 'NASDAQ_Close', 'NYSE_Close', 'RUSSELL_Close', 'SP_Close'], axis=1)

# Convert the 'Date' column to datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Extract year, month, and day as separate columns
combined_df.insert(1, 'Day', combined_df['Date'].dt.day)
combined_df.insert(1, 'Month', combined_df['Date'].dt.month)
combined_df.insert(1, 'Year', combined_df['Date'].dt.year)

# Drop the 'Date' column
combined_df = combined_df.drop(columns=['Date'])

# Save the combined dataframe to a new CSV file
combined_df.to_csv('day1prediction.csv', index=False)
