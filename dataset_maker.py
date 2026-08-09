import pandas as pd
import math
import numpy as np


#turn the csv file into a pandas dataframe
file_path = "Datasets/Top_9_S&P500_OHLCV_2010-2026.csv"

df = pd.read_csv(file_path)


#split the dataframe into seperate dataframes for each stock ticker
df_META = df.loc[df['Ticker'] == 'META'].copy()
df_GOOG = df.loc[df['Ticker'] == 'GOOG'].copy()
df_AAPL = df.loc[df['Ticker'] == 'AAPL'].copy()
df_GOOGL = df.loc[df['Ticker'] == 'GOOGL'].copy()
df_MSFT = df.loc[df['Ticker'] == 'MSFT'].copy()
df_TSLA = df.loc[df['Ticker'] == 'TSLA'].copy()
df_AVGO = df.loc[df['Ticker'] == 'AVGO'].copy()
df_NVDA = df.loc[df['Ticker'] == 'NVDA'].copy()
df_AMZN = df.loc[df['Ticker'] == 'AMZN'].copy()


#function to turn the date into the number of the week (0-6)
def get_DayOfWeek(y, m, d):

    y, m, d = int(y), int(m), int(d)
    
    full_year = 2000 + y
    is_leap = (full_year % 4 == 0)  # fine for this range, no need for /100 /400 rules

    year_code = y + (y // 4)

    month_codes = {
        1: 0 if is_leap else 1,
        2: 3 if is_leap else 4,
        3: 4,
        4: 0,
        5: 2,
        6: 5,
        7: 0,
        8: 3,
        9: 6,
        10: 1,
        11: 4,
        12: 6,
    }
    month_code = month_codes[m]

    century_code = 6    #always 6 because we are past the 2000's

    total = year_code + month_code + d + century_code
    DayOfWeek_number = total % 7

    return DayOfWeek_number


#crate a seperate dataframe for each ticker that will eventually become the training set
#adding additional columns of needen inputs for the model & getting rid of any columns that are not needed for the model, and any rows that have a NaN value (the first 29 rows)
META_training_set_df = df_META.assign(
                            YesterdayOpen = df_META['Open'].shift(1), 
                            YesterdayClose = df_META['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_META['Open'].shift(1) / df_META['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_META['High'].shift(1) / df_META['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_META['Low'].shift(1) / df_META['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_META['Volume'].shift(1) / df_META['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_META['Close'].shift(1) / df_META['Close'].shift(2)),
                            MA10 = df_META['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_META['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_META['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_META['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_META['Date'].str[8:10],
                            MonthNumber = df_META['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
GOOG_training_set_df = df_GOOG.assign(
                            YesterdayOpen = df_GOOG['Open'].shift(1), 
                            YesterdayClose = df_GOOG['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_GOOG['Open'].shift(1) / df_GOOG['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_GOOG['High'].shift(1) / df_GOOG['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_GOOG['Low'].shift(1) / df_GOOG['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_GOOG['Volume'].shift(1) / df_GOOG['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_GOOG['Close'].shift(1) / df_GOOG['Close'].shift(2)),
                            MA10 = df_GOOG['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_GOOG['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_GOOG['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_GOOG['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_GOOG['Date'].str[8:10],
                            MonthNumber = df_GOOG['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
AAPL_training_set_df = df_AAPL.assign(
                            YesterdayOpen = df_AAPL['Open'].shift(1), 
                            YesterdayClose = df_AAPL['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_AAPL['Open'].shift(1) / df_AAPL['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_AAPL['High'].shift(1) / df_AAPL['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_AAPL['Low'].shift(1) / df_AAPL['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_AAPL['Volume'].shift(1) / df_AAPL['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_AAPL['Close'].shift(1) / df_AAPL['Close'].shift(2)),
                            MA10 = df_AAPL['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_AAPL['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_AAPL['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_AAPL['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_AAPL['Date'].str[8:10],
                            MonthNumber = df_AAPL['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
GOOGL_training_set_df = df_GOOGL.assign(
                            YesterdayOpen = df_GOOGL['Open'].shift(1), 
                            YesterdayClose = df_GOOGL['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_GOOGL['Open'].shift(1) / df_GOOGL['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_GOOGL['High'].shift(1) / df_GOOGL['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_GOOGL['Low'].shift(1) / df_GOOGL['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_GOOGL['Volume'].shift(1) / df_GOOGL['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_GOOGL['Close'].shift(1) / df_GOOGL['Close'].shift(2)),
                            MA10 = df_GOOGL['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_GOOGL['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_GOOGL['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_GOOGL['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_GOOGL['Date'].str[8:10],
                            MonthNumber = df_GOOGL['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
MSFT_training_set_df = df_MSFT.assign(
                            YesterdayOpen = df_MSFT['Open'].shift(1), 
                            YesterdayClose = df_MSFT['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_MSFT['Open'].shift(1) / df_MSFT['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_MSFT['High'].shift(1) / df_MSFT['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_MSFT['Low'].shift(1) / df_MSFT['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_MSFT['Volume'].shift(1) / df_MSFT['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_MSFT['Close'].shift(1) / df_MSFT['Close'].shift(2)),
                            MA10 = df_MSFT['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_MSFT['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_MSFT['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_MSFT['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_MSFT['Date'].str[8:10],
                            MonthNumber = df_MSFT['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
TSLA_training_set_df = df_TSLA.assign(
                            YesterdayOpen = df_TSLA['Open'].shift(1), 
                            YesterdayClose = df_TSLA['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_TSLA['Open'].shift(1) / df_TSLA['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_TSLA['High'].shift(1) / df_TSLA['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_TSLA['Low'].shift(1) / df_TSLA['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_TSLA['Volume'].shift(1) / df_TSLA['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_TSLA['Close'].shift(1) / df_TSLA['Close'].shift(2)),
                            MA10 = df_TSLA['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_TSLA['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_TSLA['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_TSLA['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_TSLA['Date'].str[8:10],
                            MonthNumber = df_TSLA['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
AVGO_training_set_df = df_AVGO.assign(
                            YesterdayOpen = df_AVGO['Open'].shift(1), 
                            YesterdayClose = df_AVGO['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_AVGO['Open'].shift(1) / df_AVGO['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_AVGO['High'].shift(1) / df_AVGO['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_AVGO['Low'].shift(1) / df_AVGO['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_AVGO['Volume'].shift(1) / df_AVGO['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_AVGO['Close'].shift(1) / df_AVGO['Close'].shift(2)),
                            MA10 = df_AVGO['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_AVGO['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_AVGO['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_AVGO['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_AVGO['Date'].str[8:10],
                            MonthNumber = df_AVGO['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
NVDA_training_set_df = df_NVDA.assign(
                            YesterdayOpen = df_NVDA['Open'].shift(1), 
                            YesterdayClose = df_NVDA['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_NVDA['Open'].shift(1) / df_NVDA['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_NVDA['High'].shift(1) / df_NVDA['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_NVDA['Low'].shift(1) / df_NVDA['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_NVDA['Volume'].shift(1) / df_NVDA['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_NVDA['Close'].shift(1) / df_NVDA['Close'].shift(2)),
                            MA10 = df_NVDA['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_NVDA['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_NVDA['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_NVDA['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_NVDA['Date'].str[8:10],
                            MonthNumber = df_NVDA['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)
AMZN_training_set_df = df_AMZN.assign(
                            YesterdayOpen = df_AMZN['Open'].shift(1), 
                            YesterdayClose = df_AMZN['Close'].shift(1),
                            YesterdayOpenLogR = np.log(df_AMZN['Open'].shift(1) / df_AMZN['Open'].shift(2)),
                            YesterdayHighLogR = np.log(df_AMZN['High'].shift(1) / df_AMZN['High'].shift(2)),
                            YesterdayLowLogR = np.log(df_AMZN['Low'].shift(1) / df_AMZN['Low'].shift(2)),
                            YesterdayVolumeLogR = np.log(df_AMZN['Volume'].shift(1) / df_AMZN['Volume'].shift(2)),
                            YesterdayCloseLogR = np.log(df_AMZN['Close'].shift(1) / df_AMZN['Close'].shift(2)),
                            MA10 = df_AMZN['Close'].shift(1).rolling(window=10).mean(),
                            MA20 = df_AMZN['Close'].shift(1).rolling(window=20).mean(),
                            MA30 = df_AMZN['Close'].shift(1).rolling(window=30).mean(),
                            DayOfWeek = df_AMZN['Date'].apply(lambda date_str: get_DayOfWeek(date_str[2:4], date_str[5:7], date_str[8:10])),
                            DayOfMonth = df_AMZN['Date'].str[8:10],
                            MonthNumber = df_AMZN['Date'].str[5:7]
                            ).drop(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Adj_Close', 'Volume']).dropna().reset_index(drop=True)


#Put all of the information into python lists in the correct dimentions for model inputs

dataset_X = []
dataset_y = []

dfs = [META_training_set_df, GOOG_training_set_df, AAPL_training_set_df, GOOGL_training_set_df, MSFT_training_set_df, TSLA_training_set_df, AVGO_training_set_df, NVDA_training_set_df, AMZN_training_set_df]

def dataframe_to_dataset_list(dataframes, X_list = dataset_X, y_list = dataset_y):
    
    sequence_number = 0
    sequence = []
    temp_X_list = []
    temp_y_list = []

    for training_set_df in dataframes:
        for run in range(0, len(training_set_df)):
            if sequence_number < len(training_set_df):
                sequence.append(training_set_df.iloc[sequence_number].drop(labels=['Close']).tolist())

                if sequence_number % 20 == 0 and sequence_number != 0:
                    temp_X_list.append(sequence.copy())
                    temp_y_list.append(training_set_df['Close'][sequence_number])
                    sequence.clear()

                sequence_number += 1

        sequence_number = 0
        sequence.clear()        
        #the first inedx is not of length 20 (first index is 21 (0-20))
        del temp_y_list[0]
        del temp_X_list[0]
        
        X_list.extend(temp_X_list.copy())
        y_list.extend(temp_y_list.copy())
        temp_X_list.clear()
        temp_y_list.clear()

    return X_list, y_list

#Run the function to turn the dataframes into python list datasets
dataset_X, dataset_y = dataframe_to_dataset_list(dataframes=dfs)


#Turn the datasets into numpy arrays and store them as .npz files to be exported and loaded into the model to train and test
X = np.array(dataset_X, dtype=np.float32)
y = np.array(dataset_y, dtype=np.float32)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

np.savez("Top_9_S&P500_dataset.npz", X=X, y=y)

