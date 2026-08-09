import pandas as pd 
import numpy as np
import torch
from torch import nn
import statistics
import yfinance as yf

#Import and define LSTM model structure and paramaters
class LSTM_by_hand(nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size

    combined_size = input_size + hidden_size


    # Forget gate: decides what to throw away from the cell state
    self.forget_gate = nn.Linear(combined_size, hidden_size)

    # Input gate: decides which new values to update in the cell state
    self.input_gate = nn.Linear(combined_size, hidden_size)

    # Candidate values: proposes new information to potentially add
    self.candidate_gate = nn.Linear(combined_size, hidden_size)

    # Output gate: decides what the new hidden state should be
    self.output_gate = nn.Linear(combined_size, hidden_size)


    # Maps the final hidden state to a single predicted price
    self.output_layer = nn.Linear(hidden_size, 1)


  def step(self, x_t, h_prev, c_prev):
    combined = torch.cat((x_t, h_prev), dim=1)

    f = torch.sigmoid(self.forget_gate(combined))
    i = torch.sigmoid(self.input_gate(combined))
    g = torch.tanh(self.candidate_gate(combined))
    o = torch.sigmoid(self.output_gate(combined))

    c_new = f * c_prev + i * g
    h_new = o * torch.tanh(c_new)

    return h_new, c_new

  def forward(self, x):
    # x shape: (batch_size, seq_len, input_size)
    batch_size, seq_len, _ = x.shape

    h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
    c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

    for t in range(seq_len):
        x_t = x[:, t, :]                     # grab this time step's input, shape (batch_size, input_size)
        h_t, c_t = self.step(x_t, h_t, c_t)

    prediction = self.output_layer(h_t)   # shape: (batch_size, 1)

    return prediction

device = torch.device("cpu")
model = LSTM_by_hand(13, 128).to(device)
model.load_state_dict(torch.load("LSTM_model_1.pth", map_location=device))
model.eval()





#Function to turn the date into the number of the week (0-6)
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


#Function to get the next day of week skipping, the weekends and append it to a new row of a dataframe
def add_next_trading_day_row(df):
    df = df.copy()
    last_date = pd.to_datetime(df['Date'].iloc[-1])

    # step forward to the next weekday (Mon-Fri)
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:  # 5=Sat, 6=Sun
        next_date += pd.Timedelta(days=1)

    placeholder = {col: np.nan for col in df.columns}
    placeholder['Date'] = next_date.strftime('%Y-%m-%d')

    return pd.concat([df, pd.DataFrame([placeholder])], ignore_index=True)

#Function that makes the batch api calls to get the information from the last 50 trading days
def get_last_n_days(tickers, n=50):
    #Returns a dict of {ticker: DataFrame}, each DataFrame indexed by Date with columns Open, High, Low, Close, Volume.

    # Yahoo's 'period' param works in calendar days, not trading days, so request extra buffer (weekends/holidays) and trim to exactly n after.
    buffer_days = n + 20

    data = yf.download(
        tickers,
        period=f"{buffer_days}d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )

    result = {}
    for ticker in tickers:
        df = data[ticker].dropna().tail(n).copy()
        result[ticker] = df

    return result


#Function that runs the information from the batch api calls through the model for each stock and decides weather or not to buy
def OHLCV_to_features(tickers_dict):
    results = {}

    for ticker, df in tickers_dict.items():
        df = df.reset_index()              
        
        df = add_next_trading_day_row(df)
        
        date_series = pd.to_datetime(df['Date'])

        featured = df.assign(
            YesterdayOpen = df['Open'],
            YesterdayClose = df['Close'],
            YesterdayOpenLogR = np.log(df['Open'] / df['Open'].shift(1)),
            YesterdayHighLogR = np.log(df['High'] / df['High'].shift(1)),
            YesterdayLowLogR = np.log(df['Low'] / df['Low'].shift(1)),
            YesterdayVolumeLogR = np.log(df['Volume'] / df['Volume'].shift(1)),
            YesterdayCloseLogR = np.log(df['Close'] / df['Close'].shift(1)),
            MA10 = df['Close'].rolling(window=10).mean(),
            MA20 = df['Close'].rolling(window=20).mean(),
            MA30 = df['Close'].rolling(window=30).mean(),
            DayOfWeek = date_series.shift(-1).apply(
            lambda dt: get_DayOfWeek(str(dt.year)[2:4], f"{dt.month:02d}", f"{dt.day:02d}") if pd.notnull(dt) else np.nan),
            DayOfMonth = date_series.shift(-1).dt.day,
            MonthNumber = date_series.shift(-1).dt.month,
                
            ).drop(columns=['Close', 'Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], errors='ignore').dropna().tail(20).reset_index(drop=True)

        results[ticker] = featured
    return results


#Function to turn the dictionary of dataframes into dictionary's of lists of sizes compatable with the model
def dataframe_to_list(dataframes):
    result_dict = {}

    for ticker, df in dataframes.items():
        sequence = []
        for i in range(len(df)):
            sequence.append(df.iloc[i].tolist())
        result_dict[ticker] = sequence

    return result_dict


#Function that prints what stocks to buy and their predicted return
def buy_or_nah(features_dict, model):
   for stock, features in features_dict.items():
        with torch.no_grad():
            pred_close = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0)).item()
        
        yesterday_close = features[-1][1]
        pred_return = (((pred_close - yesterday_close) / yesterday_close) * 100)

        print(f'{stock}: predicted close = ${pred_close:.3f} | predicted return = {pred_return:.3f}%')
   
   pass   




#In the tickers list, lisnt any and all stock tickers that you want to run through the model
#The tickers below are the tickers that the model was trained/tested on
tickers = ['META', 'GOOG', 'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AVGO', 'NVDA', 'AMZN']

buy_or_nah(dataframe_to_list(OHLCV_to_features(get_last_n_days(tickers))), model)