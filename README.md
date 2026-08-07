# LSTM Stock Predictor
 
A hand-built LSTM (written from scratch in PyTorch, without `nn.LSTM`) that predicts next-day closing prices for a set of large-cap stocks, trained on historical OHLCV data.
 
> **Disclaimer:** This project was built purely as an educational exercise in understanding LSTMs, PyTorch, and time-series modeling. It is **not** financial advice, and should not be used to make real investment decisions.
 
## Quick Start
 
If you just want to see the finished program give stock return predictions, you only need `Live_Stock_predictor.py` and `LSTM_model_1.pth` (the trained model weights). Everything else in this repo is supporting/development work and isn't required to run it.
 
1. Open `Live_Stock_predictor.py`
2. Add the tickers you want predictions for to the `tickers` list at the bottom of the file
3. Run the script
**Note on timing:** the script needs to be run after the market closes on one day and before it opens the next — it predicts the *next* trading day's close using that day's just-completed data.
 
### Requirements
 
All requirements for this project can be downloaded by installing the dependencies listed in `requirements.txt`
```bash
pip install -r requirements.txt
```
## Repository Structure
 
| File | Purpose |
|---|---|
| `dataset_maker.py` | Pulls historical OHLCV data and engineers it into the feature set used for training |
| `LSTM_training_and_testing.ipynb` | Defines the hand-built LSTM architecture and trains/evaluates it |
| `backtester.py` | Backtests the trained model against historical data and compares it to buy-and-hold |
| `Live_Stock_predictor.py` | The finished product — fetches recent data live and prints next-day return predictions |
| `LSTM_model_1.pth` | Saved weights for the trained model |
 
The dataset maker, training/testing notebook, and backtester were all steps along the way to building the live predictor — they're not needed to run the final program, but I've kept them in the repo to show the actual process (data engineering → training → backtesting → deployment), and so anyone who wants to retrain, tweak, or extend the model has a starting point rather than a black box.
 
## Known Limitations
 
Being upfront about the current weak points:
 
- **The code is unorganized.** Functions are scattered across files rather than grouped into modules or classes. Cleaning this up is priority #1 for the next version.
- **The dataset format changed at every stage.** Since I built this one step at a time rather than planning the full pipeline up front, the dataset gets reshaped/re-engineered differently in almost every file to fit that file's specific needs. It should have been designed once, correctly, from `dataset_maker.py` onward.
- **`dataset_maker.py` has a lot of copy-pasted, per-ticker code.** It repeats the same feature-engineering block once for each stock instead of looping over a dictionary of ticker dataframes (which is how `Live_Stock_predictor.py` actually does it). This doesn't scale if the ticker list grows.
- **The backtest may be optimistic.** `backtester.py` evaluates the model on AMZN specifically because AMZN wasn't part of training — but the split is by *stock*, not by *time*. The model could still be picking up on general patterns shared across all 9 stocks during that same historical period, which would make it look more accurate than it really is on genuinely unseen, future data.
- **No sentiment or news data.** Predicting next-day closes from price/volume data alone leaves the model exposed to short-term sentiment-driven noise it has no way to see coming.

## Future Improvements 
**Trading logic & automation**
- Automatically schedule and execute the predicted trade rather than just printing a number
- Size each position based on the predicted return and total portfolio size
- Combine predicted returns with other signals into an actual trading strategy, so the program decides whether to buy rather than leaving that judgment call to the user

**Model & data**
- Add more features derivable from OHLCV data alone, like MACD
- Incorporate a sentiment analysis model over news/social media for market-moving context the price data can't capture
- Predict further out (e.g. a week ahead instead of one day) — short-term sentiment noise may average out over a longer horizon, making predictions better aligned with underlying trend/moving-average behavior
- Split train/test temporally rather than by ticker, to get a more honest measure of out-of-sample accuracy

**Code architecture**
- Break the scattered functions out into proper modules/classes instead of one long script per stage
- Refactor `dataset_maker.py` to loop over a ticker dictionary instead of duplicating code per stock
- Design the dataset format once, up front, so it doesn't need to be reshaped at every downstream stage

**Training process**
- Train in batches instead of on the full dataset every epoch — not critical at this dataset size, but necessary if it's scaled up
- Implement proper early stopping that checkpoints and restores the best-validation-loss model, to guard against overfitting
