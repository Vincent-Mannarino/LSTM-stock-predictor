import pandas as pd 
import numpy as np
import torch
from torch import nn
import statistics

#In order to backtest and compare the model to just buying and holding the SPY, I need day by day information that the model has not seen over a good amount of time.
#I will load in the testing set and try to analyze the data by hand to make sure that it is only one stock and the date at which it starts.

data = np.load("Top_9_S&P500_test_set.npz")
data_X = data["X"].tolist()
data_y = data["y"].tolist()


#Deleteing the indexes that are not AMZN stock
#The staring date is 2010-10-06
del data_X[:160]
del data_y[:160]


#Manipulate the X testing sets so that the model can run every day instead of every 20 days
data_X = np.array(data_X).flatten().tolist()
new_data_X = []
data_X_features = []
temp_data_X = []

for i in range(0,len(data_X),13):
  data_X_features.append(data_X[i:(i+13)].copy())

for i in range(0,len(data_X_features) - 19):
  temp_data_X.append(data_X_features[i:i+20].copy())

data_X = temp_data_X

#Manipulate the y testing sets so that the model can run every day instead of every 20 days
temp_data_y = []

for i in range(len(data_X)-1):
  temp_data_y.append(data_X_features[i + 20][1])

temp_data_y.extend(data_y[-1])

data_y = temp_data_y


#turn the lists to torch tensors 
data_X = torch.tensor(data_X)
data_y = torch.tensor(data_y, dtype=torch.float32)


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


#Run the stock information through the model and calculate the percentage gain or loss
#This method is simply to buy if the predicted outcome is a positive return

collection_of_returns = []

for i in range(len(data_y)):
    real_value = data_y[i].item()
    pred_value = model(data_X[i].unsqueeze(0)) 
    yesterday_close = data_X[i, -1, 1].item()

    if pred_value - yesterday_close >= 0:
        collection_of_returns.append(((real_value - yesterday_close) / yesterday_close) * 100)

avg_daily_return = statistics.mean(collection_of_returns)
print(avg_daily_return)

#find more accurate portfolio value
portfolio_value = float(input('What is your portfolio value?: '))
for r in collection_of_returns:
    portfolio_value *= (1 + r / 100)

print(portfolio_value)


